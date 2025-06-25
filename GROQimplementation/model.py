import streamlit as st
from typing import Generator
from groq import Groq
import boto3
from botocore.exceptions import NoCredentialsError
from botocore.exceptions import ClientError

# === Load AWS Credentials from .streamlit/secrets.toml ===
aws_access_key_id = st.secrets["aws"]["access_key"]
aws_secret_access_key = st.secrets["aws"]["secret_key"]
bucket_name = st.secrets["aws"]["bucket_name"]
region_name = st.secrets["aws"]["region"]
folder_prefix = "download-uploads2/"

st.set_page_config(page_icon="💬", layout="wide",
                   page_title="Groq Testing")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon("🤖")

st.subheader("Groq Testing App", divider="rainbow", anchor=False)

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "meta-llama/llama-4-scout-17b-16e-instruct":{"name": "Meta-Llama-4-scout-17b-16e-instruct", "tokens": 8192 , "developer": "Meta"}
}

# Layout for model selection and max_tokens slider
col1, col2 = st.columns(2)

with col1:
    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=0  # Default to llama-4-scout
    )

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = 8192

with col2:
    # Adjust max_tokens slider dynamically based on the selected model
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,  # Minimum value to allow some flexibility
        max_value=max_tokens_range,
        # Default value or max allowed if less
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
    )
   
# Uploading docs to s3 
uploaded_file = st.file_uploader(" ", type=['txt', 'pdf', 'docx', 'ppt'], label_visibility="collapsed")
if uploaded_file:
        st.session_state.messages.append({"role": "user", "content": f"📎 Uploaded file: {uploaded_file.name}"})


custom_folder = "uploads"
if uploaded_file is not None:
    file_name = f"{custom_folder}/{uploaded_file.name}"

    try:
        # Initialize S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

        # Upload file
        s3.upload_fileobj(uploaded_file, bucket_name, file_name)
        st.success(f"✅ File '{file_name}' uploaded successfully to '{bucket_name}'!")
    
    except NoCredentialsError:
        st.error("❌ AWS credentials not found.")
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")
 
# === List files in folder ===
selected_file = st.selectbox("Select files to delete")
try:
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)
    if "Contents" in response:
        files = [obj["Key"] for obj in response["Contents"] if not obj["Key"].endswith("/")]
        selected_file = st.selectbox("Select a file to delete", files)

        if st.button("Delete File"):
            try:
                s3.delete_object(Bucket=bucket_name, Key=selected_file)
                st.success(f"✅ '{selected_file}' deleted successfully.")
            except ClientError as e:
                st.error(f"❌ Delete failed: {e}")
    else:
        st.info("No files found.")
except Exception as e:
    st.error(f"❌ Failed to list files: {e}")
        
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👩‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
