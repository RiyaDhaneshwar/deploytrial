import streamlit as st
import requests
from typing import Generator
import json
from datetime import datetime
import hashlib
import role_modfication as rm
import re
import os
from dotenv import load_dotenv
import base64

load_dotenv()

#Backend API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

#Cognito Domain
COGNITO_DOMAIN = "https://ap-south-1ldsecczr5.auth.ap-south-1.amazoncognito.com"
CLIENT_ID = "2sv0anl655lqjtbvdu8m38amp6"
REDIRECT_URI = "https://killing-soldier-acres-indoor.trycloudflare.com/"

#Parse Token Fragment
def parse_token_fragment():
    js = """
    <script>
    if (window.location.hash.includes("access_token")) {
        const params = new URLSearchParams(window.location.hash.slice(1));
        const token = params.get("access_token");
        if (token) {
            window.location.href = window.location.pathname + "?token=" + token;
        }
    }
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)

#Redirect to Login
def redirect_to_login():
    auth_url = (
        f"{COGNITO_DOMAIN}/login"
        f"?response_type=code&client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&scope=email+openid+phone"
    )
    st.markdown(f'<script>window.location.href="{auth_url}";</script>', unsafe_allow_html=True)

# Set page config
st.set_page_config(page_icon="💬", layout="wide",
                   page_title="Multi-Org Groq Testing")

#  1. Initialize Session State 
if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = hashlib.md5(f"{datetime.now().isoformat()}_{os.getpid()}".encode()).hexdigest()[:8]

if "auth_completed" not in st.session_state:
    st.session_state.auth_completed = False

if "user_info" not in st.session_state:
    st.session_state.user_info = {}

if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False

# Initialize other session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "user_authenticated" not in st.session_state:
    st.session_state.user_authenticated = False

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

if "role_lookup_completed" not in st.session_state:
    st.session_state.role_lookup_completed = False

if "profile_setup_completed" not in st.session_state:
    st.session_state.profile_setup_completed = False

# Call parse_token_fragment to handle URL hash fragments
parse_token_fragment()

#  Helper Functions
def exchange_code_for_token(code):
    token_url = f"{COGNITO_DOMAIN}/oauth2/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "redirect_uri": REDIRECT_URI
    }
    client_secret = os.getenv("COGNITO_CLIENT_SECRET")
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    if client_secret:
        basic_auth = base64.b64encode(f"{CLIENT_ID}:{client_secret}".encode()).decode()
        headers["Authorization"] = f"Basic {basic_auth}"
    try:
        res = requests.post(token_url, data=data, headers=headers)
        if res.status_code == 200:
            return res.json().get("access_token")
        else:
            st.error(f"Failed to exchange code for token: {res.text}")
            return None
    except Exception as e:
        st.error(f"Error exchanging code for token: {e}")
        return None

def get_user_info(token):
    if not token:
        return {}
    url = f"{COGNITO_DOMAIN}/oauth2/userInfo"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            return res.json()
        else:
            return {}
    except Exception as e:
        return {}

# Authentication Flow (runs only once) 
query_params = st.query_params
token_from_url = query_params.get("token")
code = query_params.get("code")


if not st.session_state.get("auth_completed", False):
    if token_from_url:
        st.session_state.token = token_from_url
        st.session_state.auth_completed = True
        st.rerun()  # One-time rerun to remove token param from URL
    elif code:
        token = exchange_code_for_token(code)
        if token:
            st.session_state.token = token
            st.session_state.auth_completed = True
            st.rerun()
        else:
            st.error("Failed to exchange authorization code.")
            # Don't stop here, let the app continue
    else:
        # Don't redirect automatically, let the user see the login button
        st.info("🔐 Please log in to continue")
        # Don't call st.stop() here

# Get User Info (only once after auth) 
if st.session_state.get("auth_completed") and not st.session_state.get("user_info"):
    user_info = get_user_info(st.session_state.token)
    if user_info.get("email"):
        st.session_state.user_info = user_info
        st.session_state.is_authenticated = True
    else:
        st.warning("Invalid session. Please log in again.")
        for k in ["auth_completed", "token", "user_info", "is_authenticated"]:
            st.session_state.pop(k, None)
        st.rerun()

#  Use Stored Authentication State 
user_info = st.session_state.user_info
email = user_info.get("email", "") if user_info else ""
is_authenticated = st.session_state.is_authenticated

# 7. User-Specific Logout Function 
def logout_user():
    """Logout only the current user"""
    # Clear all authentication-related session state
    keys_to_clear = [
        "token", "user_info", "is_authenticated", "auth_completed",
        "user_authenticated", "user_profile", "role_lookup_completed",
        "profile_setup_completed", "messages", "selected_model"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🏢")

st.subheader("Multi-Organization RAG Testing App", divider="rainbow", anchor=False)

# Define the CSV path for roles
ROLES_CSV_PATH = os.path.join(os.path.dirname(__file__), "roles.csv")

def is_valid_email(email):
    """Validate email format"""
    return re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email)

# Define the three specific roles
USER_ROLES = {
    "RAG_user": {
        "name": "RAG User",
        "description": "Can query and interact with documents",
        "permissions": ["read", "query"]
    },
    "doc_owner": {
        "name": "Document Owner",
        "description": "Can upload, modify and manage documents",
        "permissions": ["read", "query", "upload", "modify", "delete"]
    },
    "RAG_admin": {
        "name": "RAG Admin", 
        "description": "Can manage RAG system and user access",
        "permissions": ["read", "query", "admin", "manage_users", "upload", "modify", "delete"]
    }
}

# Define common departments
DEPARTMENTS = [
    "Finance", "HR", "Legal", "IT", "Operations", 
    "Marketing", "Sales", "Research", "Engineering", "Other"
]

# Define model details
models = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "name": "Meta-Llama-4-scout-17b-16e-instruct", 
        "tokens": 8192, 
        "developer": "Meta"
    }
}

#Creation of s3 buckets and sanitization of folder names
def sanitize_folder_name(name):
    """Sanitize organization/department names for S3 folder structure"""
    return "".join(c for c in name if c.isalnum() or c in ('-', '_')).lower()

#Generate a unique session ID for the user
def generate_user_session_id():
    """Generate a unique session ID for the user"""
    timestamp = datetime.now().isoformat()
    return hashlib.md5(timestamp.encode()).hexdigest()[:8]

#Create S3 folder path: document-upload2/test-output/{org}/{dept}/
def create_s3_folder_structure(organization, department):
    """Create S3 folder path: document-upload2/test-output/{org}/{dept}/"""
    org_safe = sanitize_folder_name(organization)
    dept_safe = sanitize_folder_name(department)
    return f"document-upload2/test-output/{org_safe}/{dept_safe}"

#Validate upload permissions
def validate_upload_permissions(role):
    """Check if user role has upload permissions"""
    return "upload" in USER_ROLES.get(role, {}).get("permissions", [])

#Validate admin permissions
def validate_admin_permissions(role):
    """Check if user role has admin permissions"""
    return "admin" in USER_ROLES.get(role, {}).get("permissions", [])

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["🏠 Main App", "👥 Role Management"])

with tab1:
    st.sidebar.header("🔐 User Profile")
    if not is_authenticated:
        # Show a large, prominent login link in the sidebar
        auth_url = (
            f"{COGNITO_DOMAIN}/login"
            f"?response_type=code&client_id={CLIENT_ID}"
            f"&redirect_uri={REDIRECT_URI}"
            f"&scope=email+openid+phone"
        )
        st.sidebar.markdown(
            f'''
            <div style="margin-top:30px; margin-bottom:30px; text-align:center;">
                <a href="{auth_url}" style="font-size:1.5em; font-weight:bold; color:#3366cc; text-decoration:none; padding:10px 20px; border:2px solid #3366cc; border-radius:5px; display:inline-block;">🔐 Login with Cognito</a>
            </div>
            ''',
            unsafe_allow_html=True
        )
        st.sidebar.info("You must be logged in to access user features.")
    else:
        # Fill email from Cognito if available
        fetched_email = user_info.get("email", "")

        # Organization Selection
        organization = st.sidebar.text_input(
            "Organization Name:",
            placeholder="Enter your organization name",
            help="This will create a separate folder structure for your organization"
        )

        # Department Selection  
        department = st.sidebar.selectbox(
            "Department:",
            [""] + DEPARTMENTS,
            help="Select your department within the organization"
        )

        # User Email (auto-filled)
        user_email = st.sidebar.text_input(
            "User Email:",
            value=fetched_email,
            placeholder="Enter your email address",
            help="This will be used for role-based authentication",
            disabled=bool(fetched_email)  # optional: disable if fetched
        )

        # Role lookup logic - only run once per session when both org and dept are entered
        if (user_email and is_valid_email(user_email) and 
            organization and department and  # Both org and dept must be entered
            not st.session_state.role_lookup_completed):
            try:
                roles_df = rm.import_csv(ROLES_CSV_PATH)
                user_role = rm.get_user_role(roles_df, user_email)
                if user_role:
                    st.session_state.user_profile["role"] = user_role
                    st.session_state.user_authenticated = True
                    st.session_state.role_lookup_completed = True
                    st.sidebar.success(f"✅ Authenticated as {user_role}")
                else:
                    st.sidebar.warning("⚠️ Email not found in user directory")
                    st.session_state.user_authenticated = False
            except Exception as e:
                st.sidebar.error(f"❌ Error loading user roles: {e}")
                st.session_state.user_authenticated = False
        elif user_email and not is_valid_email(user_email):
            st.sidebar.error("❌ Please enter a valid email address")
            st.session_state.user_authenticated = False
        elif st.session_state.role_lookup_completed and st.session_state.user_authenticated:
            # Display existing role status
            user_role = st.session_state.user_profile.get("role")
            if user_role:
                st.sidebar.success(f"✅ Authenticated as {user_role}")
        elif user_email and is_valid_email(user_email) and (not organization or not department):
            # Show message to complete both fields
            st.sidebar.info("ℹ️ Please enter both organization and department to continue")

        # Display role permissions if authenticated
        if st.session_state.user_authenticated:
            user_role = st.session_state.user_profile.get("role")
            if user_role and isinstance(user_role, str):
                role_info = USER_ROLES.get(user_role, {})
                if role_info:
                    st.sidebar.info(f"**{role_info['name']}**\n\n{role_info['description']}")
                    permissions = ", ".join(role_info['permissions'])
                    st.sidebar.caption(f"Permissions: {permissions}")

        # Profile completion logic
        profile_complete = all([organization, department, user_email]) and st.session_state.user_authenticated

        if profile_complete and not st.session_state.profile_setup_completed:
            # Update profile with current org/dept info
            st.session_state.user_profile.update({
                "organization": organization,
                "department": department,
                "user_id": user_email,
                "session_id": generate_user_session_id()
            })
            
            st.sidebar.success("✅ Profile Complete")
            
            # Call API to load documents
            try:
                response = requests.post(API_URL, json=st.session_state.user_profile)
                if response.status_code == 200:
                    result = response.json()
                    st.success(result["message"])
                    st.info(f'Documents Loaded: {result["documents_loaded"]}')
                    st.session_state.profile_setup_completed = True
                else:
                    st.error(f"Failed: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
        elif profile_complete and st.session_state.profile_setup_completed:
            # Update profile if org/dept changed
            current_org = st.session_state.user_profile.get("organization")
            current_dept = st.session_state.user_profile.get("department")
            
            if current_org != organization or current_dept != department:
                st.session_state.user_profile.update({
                    "organization": organization,
                    "department": department
                })
                # Reset some flags to allow re-processing
                st.session_state.profile_setup_completed = False
                st.rerun()
            else:
                st.sidebar.success("✅ Profile Complete")
        elif user_email and is_valid_email(user_email) and st.session_state.user_authenticated and (not organization or not department):
            # Show message to complete both fields
            st.sidebar.warning("⚠️ Please enter both organization and department to complete your profile")
        elif user_email and is_valid_email(user_email) and (not organization or not department):
            # Show message to complete both fields
            st.sidebar.info("ℹ️ Please enter both organization and department to continue")
        else:
            st.sidebar.warning("⚠️ Complete all fields to proceed")

        # User-specific logout button
        if is_authenticated:
            st.sidebar.markdown("---")
            if st.sidebar.button("🚪 Logout", key=f"logout_{st.session_state.user_session_id}"):
                logout_user()

        # Main content area
        col1, col2 = st.columns(2)

        with col1:
            model_option = st.selectbox(
                "Choose a model:",
                options=list(models.keys()),
                format_func=lambda x: models[x]["name"],
                index=0
            )

        # Detect model change and clear chat history if model has changed
        if st.session_state.selected_model != model_option:
            st.session_state.messages = []
            st.session_state.selected_model = model_option

        max_tokens_range = 8192

        with col2:
            st.slider(
                "Max Tokens:",
                min_value=512,
                max_value=max_tokens_range,
                value=min(32768, max_tokens_range),
                step=512,
                help=f"Adjust the maximum number of tokens for the model's response. Max: {max_tokens_range}"
            )

        # Document Upload Section
        st.subheader("📄 Document Upload")

        if not st.session_state.user_authenticated:
            st.warning("🔒 Please complete your profile and authenticate with a valid email to upload documents.")
            uploaded_file = None
        elif not validate_upload_permissions(st.session_state.user_profile.get("role", "")):
            st.error(f"❌ Your role does not have upload permissions.")
            st.info("Only Document Owners can upload files. Contact your RAG Admin for access.")
            uploaded_file = None
        elif not organization or not department:
            st.warning("⚠️ Please enter both organization and department to upload documents.")
            uploaded_file = None
        else:
            folder_path = create_s3_folder_structure(organization, department)
            st.info(f"📁 Files will be uploaded to: `{folder_path}/`")

            # Ensure key is initialized
            if "file_uploader_key" not in st.session_state:
                st.session_state["file_uploader_key"] = str(datetime.now().timestamp())

            uploaded_file = st.file_uploader(
                "Choose a file to upload:",
                type=['txt', 'pdf', 'docx', 'ppt', 'pptx'],
                key=st.session_state["file_uploader_key"]
            )

            # Upload logic — only run if we haven't already processed this upload. To do- if hash is same but file is different, then upload again.
            if uploaded_file and not st.session_state.get("file_uploaded_once", False):
                try:
                    with open(uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    payload = {
                        "file_path": uploaded_file.name,
                        "base_output_dir": "document-upload2/test-output",
                        "org": organization,
                        "dept": department,
                        "user_email": user_email,
                        "debug_mode": True
                    }

                    response = requests.post(f"{API_URL}/upload_docs", json=payload)
                    if response.status_code == 200:
                        st.success("✅ File uploaded and processed successfully!")
                        st.json(response.json())

                        if "deleted_documents" in st.session_state:
                            st.session_state.deleted_documents = set()

                        st.session_state["file_uploaded_once"] = True
                        st.session_state["file_uploader_key"] = str(datetime.now().timestamp())  # Force refresh
                        
                        # Clear document cache to force refresh
                        keys_to_clear = [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith("doc_list_")]
                        for key in keys_to_clear:
                            del st.session_state[key]
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Upload failed: {e}")
                finally:
                    if os.path.exists(uploaded_file.name):
                        os.remove(uploaded_file.name)

        # Reset the upload flag when there's no uploaded file
        if st.session_state.get("file_uploaded_once", False) and not uploaded_file:
            del st.session_state["file_uploaded_once"]

        #  Document Management Section 
        st.subheader("🗂️ Document Management")

        if not st.session_state.user_authenticated:
            st.warning("🔒 Please complete your profile and authenticate to manage documents.")
        elif not validate_upload_permissions(st.session_state.user_profile.get("role", "")):
            st.error("❌ You don't have permission to manage documents.")
        elif not organization or not department:
            st.warning("⚠️ Please enter both organization and department to manage documents.")
        else:
            if "deleted_documents" not in st.session_state:
                st.session_state.deleted_documents = set()

            # Clear deleted documents when org/dept changes
            current_context = f"{organization}_{department}"
            if "current_org_dept" not in st.session_state:
                st.session_state.current_org_dept = current_context
            elif st.session_state.current_org_dept != current_context:
                st.session_state.deleted_documents = set()
                st.session_state.current_org_dept = current_context
                # Clear document cache
                keys_to_clear = [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith("doc_list_")]
                for key in keys_to_clear:
                    del st.session_state[key]

            try:
                # Create a unique key for this org/dept combination
                doc_list_key = f"doc_list_{organization}_{department}"
                
                # Only fetch documents if we haven't already
                if doc_list_key not in st.session_state:
                    params = {
                        "org": organization,
                        "dept": department,
                        "user_email": user_email,
                        "user_role": st.session_state.user_profile.get("role"),
                        "base_output_dir": "document-upload2/test-output"
                    }
                    response = requests.get(f"{API_URL}/list_documents", params=params)

                    if response.status_code == 200:
                        data = response.json()
                        documents = data.get("documents", [])
                        st.session_state[doc_list_key] = documents
                    else:
                        st.error(f"❌ Failed to fetch documents: {response.text}")
                        documents = []
                else:
                    documents = st.session_state[doc_list_key]

                active_documents = [doc for doc in documents if doc['doc_id'] not in st.session_state.deleted_documents]

                if active_documents: #If there are active documents, then show them as per the user role.
                    user_role = st.session_state.user_profile.get("role")
                    if user_role == "doc_owner":
                        st.info(f"👤 Showing only your documents (Role: Document Owner)")
                    elif user_role == "RAG_admin":
                        st.info(f"👑 Showing all documents in {organization}/{department} (Role: RAG Admin)")
                    elif user_role == "RAG_user":
                        st.info(f"👥 Showing all documents for querying (Role: RAG User)")

                    st.write(f"**Found {len(active_documents)} document(s):**")

                    for doc in active_documents:
                        if doc['doc_id'] in st.session_state.deleted_documents:
                            continue

                        ownership_info = f" (by {doc['uploaded_by']})" if doc.get('uploaded_by') else ""

                        with st.expander(f"📄 {doc['doc_id']}{ownership_info} ({doc['text_chunks']} text chunks, {doc['images']} images)"):
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                st.write(f"**Document ID:** {doc['doc_id']}")
                                st.write(f"**Text Chunks:** {doc['text_chunks']}")
                                st.write(f"**Images:** {doc['images']}")
                                if doc['upload_date']:
                                    st.write(f"**Upload Date:** {doc['upload_date']}")
                                if doc.get('uploaded_by'):
                                    st.write(f"**Uploaded By:** {doc['uploaded_by']}")

                            with col2:
                                delete_key = f"delete_btn_{doc['doc_id']}"
                                if st.button(f"🗑️ Delete", key=delete_key):
                                    try:
                                        delete_response = requests.delete(
                                            f"{API_URL}/delete_document",
                                            params={
                                                "org": organization,
                                                "dept": department,
                                                "doc_id": doc['doc_id'],
                                                "base_output_dir": "document-upload2/test-output"
                                            }
                                        )

                                        if delete_response.status_code == 200:
                                            st.session_state.deleted_documents.add(doc['doc_id'])
                                            st.success(f"✅ Document {doc['doc_id']} deleted successfully!")
                                            # Clear document cache to force refresh
                                            if doc_list_key in st.session_state:
                                                del st.session_state[doc_list_key]
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Failed to delete document: {delete_response.text}")
                                    except Exception as e:
                                        st.error(f"❌ Error deleting document: {e}")
                else:
                    st.info("📭 No documents found for this organization and department.")
            except Exception as e:
                st.error(f"❌ Error managing documents: {e}")

        #  Chat Interface (calls FastAPI backend) 
        st.subheader("💬 Chat Interface")

        for message in st.session_state.messages:
            avatar = '🤖' if message["role"] == "assistant" else '👩‍💻'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if not st.session_state.user_authenticated:
            st.info("🔒 Complete your profile and authenticate to start chatting.")
        elif not organization or not department:
            st.info("ℹ️ Please enter both organization and department to start chatting.")
        else:
            if prompt := st.chat_input("Enter your prompt here..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar='👨‍💻'):
                    st.markdown(prompt)
                # Call backend for answer
                try:
                    response = requests.post(f"{API_URL}/Answer", json={
                        "query": prompt,
                        "org": st.session_state.user_profile["organization"],
                        "dept": st.session_state.user_profile["department"]
                    })
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("result", "No answer returned.")
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(answer)

                        # Display Source documents: Name, Location and Download link.
                        if "source_documents" in data:
                            st.markdown("## 📂 Source Documents")
                            for doc_key in data["source_documents"]:
                                parts = doc_key.split("/")
                                doc_id = parts[-2] if len(parts) >= 2 else doc_key
                                try:
                                    doc_response = requests.get(
                                        f"{API_URL}/get_source_document/{doc_id}",
                                        params={
                                            "org": st.session_state.user_profile["organization"],
                                            "dept": st.session_state.user_profile["department"]
                                        }
                                    )

                                    if doc_response.status_code == 200:
                                        doc_data = doc_response.json()
                                        st.markdown(f"**Document Name:** `{doc_data['doc_id']}`")
                                        st.markdown(f"**S3 Storage Location:** `{doc_data['s3_prefix']}`")
                                        if doc_data.get('original_file'):
                                            orig = doc_data['original_file']
                                            st.markdown(f"**Original File:** `{orig.get('filename', 'N/A')}`")
                                            st.markdown(f"**Size:** {orig.get('size', 'N/A')} bytes")
                                            st.markdown(f"**Content Type:** {orig.get('content_type', 'N/A')}")
                                            if orig.get('download_url'):
                                                st.markdown(f"[Download Original File]({orig['download_url']})")
                                            elif orig.get('error'):
                                                st.warning(f"Error loading original file: {orig['error']}")
                                    else:
                                        st.warning(f"⚠️ Could not load content for `{doc_id}`: {doc_response.text}")
                                except Exception as fetch_err:
                                    st.error(f"❌ Error fetching document `{doc_id}`: {fetch_err}")
                        else:
                            st.error(f"❌ Error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        # Footer with current session info
        if st.session_state.user_authenticated:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Current Session:**")
            profile = st.session_state.user_profile
            st.sidebar.caption(f"""
            🏢 {profile.get('organization', 'Not set')}  
            🏬 {profile.get('department', 'Not set')}  
            👤 {profile.get('user_id', 'Not set')}  
            🎭 {USER_ROLES.get(profile.get('role', ''), {}).get('name', 'Unknown')}
            """)

#Role Management Tab: for RAG Admin to manage users and their roles.
with tab2:
    st.subheader("👥 Role Management")
    # Check if user has admin permissions
    if not is_authenticated:
        st.warning("🔒 Please log in first to access role management.")
    elif not st.session_state.user_authenticated:
        st.warning("🔒 Please complete your profile first to access role management.")
    elif not validate_admin_permissions(st.session_state.user_profile.get("role", "")):
        st.error("❌ You don't have permission to manage roles. Admin access required.")
    else:
        st.success(f"✅ Welcome, {USER_ROLES.get(st.session_state.user_profile.get('role', ''), {}).get('name', 'Admin')}!")
        # Load roles from CSV
        try:
            df = rm.import_csv(ROLES_CSV_PATH)
            operation = st.selectbox("Select an operation", ["Add", "Update", "Delete"])
            operation_performed = False
            
            if operation == "Add":
                st.write("Add a new user")
                user_name = st.text_input("Enter the user name")
                user_organization = st.text_input("Enter the organization")
                user_email = st.text_input("Enter the user email")
                user_role = st.selectbox("Select the user role", options=["RAG_admin", "doc_owner", "RAG_user"])
                if st.button("Add User"):
                    if not user_name or not user_organization or not user_email:
                        st.error("Please fill in all fields.")
                    elif not is_valid_email(user_email):
                        st.error("Please enter a valid user email address.")
                    elif rm.email_exists(df, user_email):
                        st.warning("User already exists.")
                    else:
                        new_user = {"name": user_name, "organization": user_organization, "email": user_email, "role": user_role}
                        df = rm.add_row(df, new_user)
                        rm.save_csv(df, ROLES_CSV_PATH)
                        st.success(f"Added user {user_name} ({user_email}) with role {user_role}.")
                        operation_performed = True
            elif operation == "Update":
                st.write("Update a user")
                emails = df['email'].tolist()
                if emails:
                    selected_email = st.selectbox('Select email to update', emails)
                    new_role = st.selectbox("Select the new role", options=["RAG_admin", "doc_owner", "RAG_user"])
                    if st.button("Update Role"):
                        df, success = rm.update_row_by_email(df, selected_email, new_role)
                        if success:
                            rm.save_csv(df, ROLES_CSV_PATH)
                            st.success(f"Role updated for {selected_email} to {new_role}.")
                            operation_performed = True
                        else:
                            st.warning("User does not exist.")
                else:
                    st.info('No users to update.')
            elif operation == "Delete":
                st.write("Delete a user")
                emails = df['email'].tolist()
                if emails:
                    selected_email = st.selectbox('Select email to delete', emails)
                    if st.button("Delete User"):
                        df, success = rm.delete_row_by_email(df, selected_email)
                        if success:
                            rm.save_csv(df, ROLES_CSV_PATH)
                            st.success(f"Deleted user: {selected_email}")
                            operation_performed = True
                        else:
                            st.warning("User does not exist.")
                else:
                    st.info('No users to delete.')
            if not operation_performed:
                st.subheader("Current Users")
                st.write(df)
            if operation_performed:
                st.subheader("Updated Users")
                st.write(df)
        except Exception as e:
            st.error(f"Error loading user roles: {e}")
