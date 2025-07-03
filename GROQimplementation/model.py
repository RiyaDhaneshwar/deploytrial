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

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_icon="💬", layout="wide",
                   page_title="Multi-Org Groq Testing")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🏢")

st.subheader("Multi-Organization RAG Testing App", divider="rainbow", anchor=False)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "user_authenticated" not in st.session_state:
    st.session_state.user_authenticated = False

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

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

def sanitize_folder_name(name):
    """Sanitize organization/department names for S3 folder structure"""
    return "".join(c for c in name if c.isalnum() or c in ('-', '_')).lower()

def generate_user_session_id():
    """Generate a unique session ID for the user"""
    timestamp = datetime.now().isoformat()
    return hashlib.md5(timestamp.encode()).hexdigest()[:8]

def create_s3_folder_structure(organization, department):
    """Create S3 folder path: document-upload2/test-output/{org}/{dept}/"""
    org_safe = sanitize_folder_name(organization)
    dept_safe = sanitize_folder_name(department)
    return f"document-upload2/test-output/{org_safe}/{dept_safe}"

def validate_upload_permissions(role):
    """Check if user role has upload permissions"""
    return "upload" in USER_ROLES.get(role, {}).get("permissions", [])

def validate_admin_permissions(role):
    """Check if user role has admin permissions"""
    return "admin" in USER_ROLES.get(role, {}).get("permissions", [])

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["🏠 Main App", "👥 Role Management"])

with tab1:
    # Sidebar for User Authentication/Profile
    st.sidebar.header("🔐 User Profile")

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

    # User ID/Email for authentication
    user_email = st.sidebar.text_input(
        "User Email:",
        placeholder="Enter your email address",
        help="This will be used for role-based authentication"
    )

    # Load roles from CSV
    try:
        roles_df = rm.import_csv(ROLES_CSV_PATH)
        if user_email and is_valid_email(user_email):
            user_role = rm.get_user_role(roles_df, user_email)
            if user_role:
                st.sidebar.success(f"✅ Authenticated as {user_role}")
                st.session_state.user_authenticated = True
                st.session_state.user_profile = {
                    "organization": organization,
                    "department": department,
                    "role": user_role,
                    "user_id": user_email,
                    "session_id": generate_user_session_id()
                }
            else:
                st.sidebar.warning("⚠️ Email not found in user directory")
                st.session_state.user_authenticated = False
        elif user_email and not is_valid_email(user_email):
            st.sidebar.error("❌ Please enter a valid email address")
            st.session_state.user_authenticated = False
        else:
            st.session_state.user_authenticated = False
    except Exception as e:
        st.sidebar.error(f"❌ Error loading user roles: {e}")
        st.session_state.user_authenticated = False

    # Display role permissions if authenticated
    if st.session_state.user_authenticated and user_email:
        user_role = st.session_state.user_profile["role"]
        role_info = USER_ROLES.get(user_role, {})
        if role_info:
            st.sidebar.info(f"**{role_info['name']}**\n\n{role_info['description']}")
            permissions = ", ".join(role_info['permissions'])
            st.sidebar.caption(f"Permissions: {permissions}")

    # Validate user profile completeness
    profile_complete = all([organization, department, user_email]) and st.session_state.user_authenticated

    if profile_complete:
        st.sidebar.success("✅ Profile Complete")
        try:
            response = requests.post(API_URL, json=st.session_state.user_profile)
            if response.status_code == 200:
                result = response.json()
                st.success(result["message"])
                st.info(f'Documents Loaded: {result["documents_loaded"]}')
            else:
                st.error(f"Failed: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
    else:
        st.sidebar.warning("⚠️ Complete all fields to proceed")

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
        max_tokens = st.slider(
            "Max Tokens:",
            min_value=512,
            max_value=max_tokens_range,
            value=min(32768, max_tokens_range),
            step=512,
            help=f"Adjust the maximum number of tokens for the model's response. Max: {max_tokens_range}"
        )

    # --- File Upload Section (calls FastAPI backend) ---
    st.subheader("📄 Document Upload")

    if not st.session_state.user_authenticated:
        st.warning("🔒 Please complete your profile and authenticate with a valid email to upload documents.")
        uploaded_file = None
    elif not validate_upload_permissions(st.session_state.user_profile.get("role", "")):
        st.error(f"❌ Your role ({USER_ROLES.get(st.session_state.user_profile.get('role', ''), {}).get('name', 'Unknown')}) does not have upload permissions.")
        st.info("Only Document Owners can upload files. Contact your RAG Admin for access.")
        uploaded_file = None
    else:
        # Show current upload location
        folder_path = create_s3_folder_structure(organization, department)
        st.info(f"📁 Files will be uploaded to: `{folder_path}/`")
        uploaded_file = st.file_uploader(
            "Choose a file to upload:",
            type=['txt', 'pdf', 'docx', 'ppt', 'pptx'],
            help="Supported formats: TXT, PDF, DOCX, PPT, PPTX"
        )

    # Handle file upload via API
    if uploaded_file and st.session_state.user_authenticated:
        try:
            # Save file temporarily
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Prepare payload
            payload = {
                "file_path": uploaded_file.name,
                "base_output_dir": "document-upload2/test-output",
                "org": organization,
                "dept": department,
                "debug_mode": True
            }
            # Call backend
            response = requests.post(f"{API_URL}/upload_docs", json=payload)
            if response.status_code == 200:
                st.success(f"✅ File uploaded and processed successfully!")
                st.json(response.json())
                # Clear deleted documents cache and refresh list after successful upload
                if "deleted_documents" in st.session_state:
                    st.session_state.deleted_documents = set()
                st.rerun()
            else:
                st.error(f"❌ Upload failed: {response.text}")
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")
        finally:
            if os.path.exists(uploaded_file.name):
                os.remove(uploaded_file.name)

    # --- Document Management Section ---
    st.subheader("🗂️ Document Management")
    
    if not st.session_state.user_authenticated:
        st.warning("🔒 Please complete your profile and authenticate to manage documents.")
    elif not validate_upload_permissions(st.session_state.user_profile.get("role", "")):
        st.error("❌ You don't have permission to manage documents.")
    else:
        # Initialize deleted documents tracking in session state
        if "deleted_documents" not in st.session_state:
            st.session_state.deleted_documents = set()
        
        # Clear deleted documents if organization or department changes
        if "current_org_dept" not in st.session_state:
            st.session_state.current_org_dept = f"{organization}_{department}"
        elif st.session_state.current_org_dept != f"{organization}_{department}":
            st.session_state.deleted_documents = set()
            st.session_state.current_org_dept = f"{organization}_{department}"
        
        # List documents
        try:
            params = {
                "org": organization,
                "dept": department,
                "base_output_dir": "document-upload2/test-output"
            }
            response = requests.get(f"{API_URL}/list_documents", params=params)
            
            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", [])
                
                # Filter out deleted documents
                active_documents = [doc for doc in documents if doc['doc_id'] not in st.session_state.deleted_documents]
                
                if active_documents:
                    st.write(f"**Found {len(active_documents)} document(s):**")
                    
                    # Create a container for documents
                    doc_container = st.container()
                    
                    for doc in active_documents:
                        # Check if this document was just deleted
                        if doc['doc_id'] in st.session_state.deleted_documents:
                            continue  # Skip this document if it was deleted
                            
                        with doc_container.expander(f"📄 {doc['doc_id']} ({doc['text_chunks']} text chunks, {doc['images']} images)"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Document ID:** {doc['doc_id']}")
                                st.write(f"**Text Chunks:** {doc['text_chunks']}")
                                st.write(f"**Images:** {doc['images']}")
                                if doc['upload_date']:
                                    st.write(f"**Upload Date:** {doc['upload_date']}")
                            
                            with col2:
                                # Use a unique key for each delete button
                                delete_key = f"delete_btn_{doc['doc_id']}"
                                if st.button(f"🗑️ Delete", key=delete_key):
                                    # Perform the actual deletion
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
                                            # Add to deleted documents set
                                            st.session_state.deleted_documents.add(doc['doc_id'])
                                            st.success(f"✅ Document {doc['doc_id']} deleted successfully!")
                                            # Force rerun to update the list immediately
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Failed to delete document: {delete_response.text}")
                                    except Exception as e:
                                        st.error(f"❌ Error deleting document: {e}")
                else:
                    st.info("📭 No documents found for this organization and department.")
                    
            else:
                st.error(f"❌ Failed to fetch documents: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Error managing documents: {e}")

    # --- Chat Interface (calls FastAPI backend) ---
    st.subheader("💬 Chat Interface")

    for message in st.session_state.messages:
        avatar = '🤖' if message["role"] == "assistant" else '👩‍💻'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if not st.session_state.user_authenticated:
        st.info("🔒 Complete your profile and authenticate to start chatting.")
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

                    # Display relevant documents
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
                                    st.markdown(f"### 📄 Document: `{doc_data['doc_id']}`")

                                    # Text chunks
                                    if doc_data.get("text_chunks"):
                                        st.markdown("#### 📝 Text Chunks")
                                        for txt in doc_data["text_chunks"]:
                                            st.code(txt["content"])

                                    # Image chunks
                                    if doc_data.get("images"):
                                        st.markdown("#### 🖼️ Images")
                                        for img in doc_data["images"]:
                                            st.markdown(f"**{img['key']}**")
                                            try:
                                                st.image(base64.b64decode(img["base64"]), use_column_width=True)
                                            except Exception as decode_err:
                                                st.warning(f"⚠️ Failed to decode image: {decode_err}")
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
        🏢 {profile['organization']}  
        🏬 {profile['department']}  
        👤 {profile['user_id']}  
        🎭 {USER_ROLES.get(profile['role'], {}).get('name', 'Unknown')}
        """)

with tab2:
    st.subheader("👥 Role Management")
    # Check if user has admin permissions
    if not st.session_state.user_authenticated:
        st.warning("🔒 Please authenticate first to access role management.")
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
