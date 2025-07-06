import os
import shutil
import zipfile
import base64
import hashlib
import pickle
import numpy as np
import faiss
import torch
import boto3
from pathlib import Path
from typing import List, Callable, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from better_profanity import profanity
from dotenv import load_dotenv
from groq import Groq
from transformers import BertModel, BertTokenizer
from sentence_transformers import util
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.text import partition_text
from fastapi.middleware.cors import CORSMiddleware
from botocore.exceptions import ClientError
import mimetypes

# Load environment variables
load_dotenv()

# AWS and Groq setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")
if not S3_BUCKET:
    raise ValueError("bucket_name environment variable is required")
if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("AWS credentials (access_key, secret_key) are required")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)
groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
profanity.load_censor_words()

bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Globals with metadata persistence
docs, doc_keys, doc_ids, faiss_index = [], [], [], None

# Metadata persistence
METADATA_PATH = "metadata_store.pkl"

def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_metadata(db):
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(db, f)

metadata_db = load_metadata()

def purge_stale_metadata():
    """Remove metadata entries for files that no longer exist in S3"""
    keys_to_delete = []
    for key in metadata_db:
        try:
            parts = key.split("/")
            if len(parts) != 3:
                continue
            org, dept, filename = parts
            prefix = f"document-upload2/test-output/{org}/{dept}/{filename}/"
            resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
            if "Contents" not in resp or not resp["Contents"]:
                keys_to_delete.append(key)
        except Exception as e:
            print(f"[Startup Check Error] Failed on key {key}: {e}")
    
    for k in keys_to_delete:
        del metadata_db[k]
        print(f"[Startup Cleanup] Removed stale metadata: {k}")
    
    if keys_to_delete:
        save_metadata(metadata_db)

# Run startup cleanup
purge_stale_metadata()

def compute_file_sha256(file_path: str) -> str:
    """Compute SHA256 hash of a file"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def encode_documents(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.array([]).reshape(0, 768)
    inputs = bert_tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

def set_up_faiss_index(documents: list[str], encode_fn: Callable) -> faiss.Index:
    index = faiss.IndexFlatL2(768)
    if documents:
        embeddings = encode_fn(documents).astype(np.float32)
        if embeddings.size > 0:
            embeddings = np.atleast_2d(embeddings)
            index.add(embeddings)
    return index

def load_documents_from_s3(prefix: str) -> list[tuple[str, str, str]]:
    """Load documents from S3 with improved structure"""
    documents = []
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith(".txt"):
                file_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                content = file_obj['Body'].read().decode("utf-8")
                doc_id = Path(key).parts[-3]
                documents.append((doc_id, key, content))
    except Exception as e:
        print(f"S3 Load Error: {e}")
    return documents

def reload_index_s3(prefix: str):
    """Reload FAISS index from S3 documents"""
    global docs, doc_keys, doc_ids, faiss_index
    doc_triples = load_documents_from_s3(prefix)
    docs = [content for _, _, content in doc_triples]
    doc_keys = [key for _, key, _ in doc_triples]
    doc_ids = [doc_id for doc_id, _, _ in doc_triples]
    
    # Build FAISS index
    embeddings = encode_documents(docs).astype(np.float32)
    if embeddings.size > 0:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss_index = index
    else:
        faiss_index = faiss.IndexFlatL2(768)

def ensure_index_built(prefix: str) -> bool:
    global faiss_index
    if faiss_index is not None and faiss_index.ntotal > 0:
        return True
    try:
        reload_index_s3(prefix)
        return faiss_index is not None and faiss_index.ntotal > 0
    except Exception as e:
        print(f"[ensure_index_built] Failed to build FAISS index: {e}")
        return False

def is_similar(embedding1, embedding2, threshold=0.8):
    """Check if two embeddings are similar using cosine similarity"""
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return similarity.item() >= threshold

def retrieve_similar_documents(query: str, index, documents: list[str], encode_fn, k=5, threshold=500):
    """Retrieve similar documents using L2 distance. Lower distance means more similar. Threshold is the maximum L2 distance (default 500)."""
    if index is None or not documents or index.ntotal == 0:
        return []
    try:
        emb = encode_fn([query]).astype(np.float32)
        emb = np.atleast_2d(emb)
        k = min(max(1, k), len(documents), index.ntotal)
        distances, indices = index.search(emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(documents) and dist <= threshold:
                results.append({
                    "content": documents[idx],
                    "score": float(dist),
                    "s3_key": doc_keys[idx],
                    "doc_id": doc_ids[idx]
                })
        return results
    except Exception as e:
        print(f"Error in retrieve_similar_documents: {e}")
        return []

def format_prompt(query: str, context: str) -> str:
    return f"""Use the following pieces of context to answer the question at the end.\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""

def generate_answer(query: str, docs_with_scores: list[tuple[str, float]]) -> str:
    if not docs_with_scores:
        return "No relevant documents found to answer the question."
    context = "\n".join(doc for doc, _ in docs_with_scores)
    prompt = format_prompt(query, context)
    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            top_p=1
        )
        return response.choices[0].message.content.strip() if response.choices else "No answer generated."
    except Exception as e:
        return f"Error generating response: {str(e)}"

class Query(BaseModel):
    query: str
    org: str
    dept: str

class UploadRequest(BaseModel):
    file_path: str
    base_output_dir: str
    org: str
    dept: str
    user_email: Optional[str] = None  # Track who uploaded the document
    debug_mode: bool = False

@app.post("/")
def auto_build_index(profile: dict):
    org = profile.get("organization")
    dept = profile.get("department")
    base_output_dir = "document-upload2/test-output"
    prefix = f"{base_output_dir}/{org}/{dept}"
    if not org or not dept:
        raise HTTPException(status_code=400, detail="Missing organization or department")
    built = ensure_index_built(prefix)
    return {
        "message": "Index is ready" if built else "Failed to build index",
        "index_built": built,
        "documents_loaded": len(docs)
    }

def upload_to_s3(key: str, data: bytes, content_type: str):
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)
        print(f"[Upload] {key} ({content_type}) - {len(data)} bytes")
    except Exception as e:
        print(f"[Upload Error] {key}: {e}")
        raise

@app.post("/upload_docs")
def process_file(data: UploadRequest) -> List[str]:
    def extract_images_from_zip(file_path: str, zip_prefix: str, output_dir: str) -> List[str]:
        images = []
        try:
            with zipfile.ZipFile(file_path, 'r') as z:
                media_files = [f for f in z.namelist() if f.startswith(zip_prefix)]
                os.makedirs(output_dir, exist_ok=True)
                for i, media_file in enumerate(media_files):
                    zip_data = z.read(media_file)
                    ext = Path(media_file).suffix or ".png"
                    path = os.path.join(output_dir, f"image_{i}{ext}")
                    with open(path, "wb") as f:
                        f.write(zip_data)
                    images.append(path)
        except Exception as e:
            print(f"[Zip Extraction Error] {e}")
        return images

    file_path = data.file_path
    ext = Path(file_path).suffix.lower()
    file_stem = Path(file_path).stem
    s3_prefix = f"{data.base_output_dir}/{data.org}/{data.dept}/{file_stem}/"
    
    # Check for duplicate file using SHA256 hash
    file_sha256 = compute_file_sha256(file_path)
    meta_key = f"{data.org}/{data.dept}/{file_stem}"
    
    if meta_key in metadata_db:
        stored_hash = metadata_db[meta_key].get("sha256")
        if stored_hash == file_sha256:
            if data.debug_mode:
                print(f"[Duplicate Detection] File {file_stem} already processed with same content")
            return [f"Duplicate File detected {file_stem}"]  # Skip processing duplicate file
    
    text_chunks = []
    
    # Upload original file to S3
    try:
        with open(file_path, "rb") as f:
            original_file_data = f.read()
            # Determine content type based on file extension
            content_type_map = {
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ".txt": "text/plain"
            }
            content_type = content_type_map.get(ext, "application/octet-stream")
            original_file_key = s3_prefix + f"original_file/{file_stem}{ext}"
            upload_to_s3(original_file_key, original_file_data, content_type)
            if data.debug_mode:
                print(f"[Info] Uploaded original file: {original_file_key}")
    except Exception as e:
        if data.debug_mode:
            print(f"[Original File Upload Error] {file_path}: {e}")
    
    image_paths = extract_images_from_zip(file_path, "word/media/" if ext == ".docx" else "ppt/media/", "/tmp") if ext in [".docx", ".pptx"] else []

    # Upload images extracted from zip (for DOCX or PPTX)
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as f:
                upload_to_s3(s3_prefix + "images/" + Path(img_path).name, f.read(), "image/png")
        except Exception as e:
            if data.debug_mode:
                print(f"[Image Upload Error] {img_path}: {e}")

    try:
        if ext == ".docx":
            elements = partition_docx(filename=file_path, infer_table_structure=True)
        elif ext == ".pptx":
            elements = partition_pptx(filename=file_path)
        elif ext == ".txt":
            elements = partition_text(filename=file_path)
        elif ext == ".pdf":
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                extract_images=True
            )

            # Upload PDF-extracted images
            for i, el in enumerate(elements):
                image_path = getattr(el.metadata, "image_path", None)
                if image_path:
                    if os.path.exists(image_path):
                        try:
                            with open(image_path, "rb") as f:
                                image_data = f.read()
                                s3_key = s3_prefix + f"images/pdf_image_{i}.png"
                                upload_to_s3(s3_key, image_data, "image/png")
                        except Exception as img_err:
                            if data.debug_mode:
                                print(f"Image upload error for {image_path}: {img_err}")
                    else:
                        if data.debug_mode:
                            print(f"[Warning] Image path not found on disk: {image_path}")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        if data.debug_mode:
            print(f"[Partitioning Error] {e}")
        return []

    # Upload a .keep file if no images were uploaded
    if not image_paths and ext in [".pdf", ".docx", ".pptx"]:
        try:
            dummy_key = s3_prefix + "images/.keep"
            s3.put_object(Bucket=S3_BUCKET, Key=dummy_key, Body=b"", ContentType="text/plain")
            if data.debug_mode:
                print(f"[Info] Created placeholder image folder: {dummy_key}")
        except Exception as e:
            if data.debug_mode:
                print(f"[Warning] Could not create placeholder folder: {e}")

    # Process and upload text chunks
    content_embeddings = []
    for i, el in enumerate(elements):
        if el.text and el.text.strip():
            try:
                text = el.text.encode("utf-8")
                key = s3_prefix + f"text/{file_stem}{i}.txt"
                upload_to_s3(key, text, "text/plain")
                text_chunks.append(key)
                
                # Generate embedding for content similarity checking
                content_embedding = encode_documents([el.text.strip()])
                content_embeddings.append(content_embedding)
                
            except Exception as text_err:
                if data.debug_mode:
                    print(f"[Text Upload Error] {text_err}")

    # Store metadata with SHA256 hash and embeddings
    if content_embeddings:
        # Average all content embeddings for the file
        avg_embedding = np.mean(content_embeddings, axis=0)
        metadata_db[meta_key] = {
            "sha256": file_sha256,
            "embedding": avg_embedding.tolist()  # Convert to list for JSON serialization
        }
        save_metadata(metadata_db)

    # Rebuild the FAISS index after successful upload
    try:
        org_dept_prefix = f"{data.base_output_dir}/{data.org}/{data.dept}"
        reload_index_s3(org_dept_prefix)
        if data.debug_mode:
            print(f"[Info] Rebuilt FAISS index for prefix: {org_dept_prefix}")
    except Exception as e:
        if data.debug_mode:
            print(f"[Warning] Failed to rebuild index: {e}")

    return text_chunks

@app.get("/index_status")
def get_index_status(org: str, dept: str, base_output_dir: str = "document-upload2/test-output"):
    prefix = f"{base_output_dir}/{org}/{dept}"
    ensure_index_built(prefix)
    return {
        "index_built": faiss_index is not None,
        "total_documents": len(docs),
        "index_size": faiss_index.ntotal if faiss_index else 0,
        "document_ids": doc_ids,
        "metadata_entries": len(metadata_db)
    }

@app.post("/Answer")
def answer_question(data: Query):
    global faiss_index

    prefix = f"document-upload2/test-output/{data.org}/{data.dept}"

    # Rebuild index if it's not already built or is empty
    if faiss_index is None or faiss_index.ntotal == 0:
        try:
            reload_index_s3(prefix)
            if faiss_index is None or faiss_index.ntotal == 0:
                return {"result": "No Relevant Document Found", "source_documents": []}
        except Exception as e:
            return {"result": f"Error rebuilding index: {str(e)}", "source_documents": []}

    # Retrieve relevant documents
    retrieved = retrieve_similar_documents(
        data.query,
        faiss_index,
        docs,
        encode_documents,
        k=5,
        threshold=500
    )

    if retrieved:
         # Generate answer from retrieved docs
        answer = generate_answer(data.query, [(r["content"], r["score"]) for r in retrieved])
        full_docs = list({r["doc_id"] for r in retrieved})  # Unique document IDs

        return {
            "result": answer,
            "source_documents": full_docs
        }
    else:
        return {"result": "No Relevant Document Found", "source_documents": []}

@app.get("/get_source_document/{doc_id}")
def get_source_document(doc_id: str, org: str, dept: str, base_output_dir: str = "document-upload2/test-output"):
    prefix = f"{base_output_dir}/{org}/{dept}/{doc_id}/"

    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        if "Contents" not in response:
            raise HTTPException(status_code=404, detail=f"No document found in S3 at prefix: {prefix}")

        original_file = None
        original_file_url = None

        for obj in response["Contents"]:
            key = obj["Key"]

            # Look for the original file
            if "/original_file/" in key and key.lower().endswith((".pdf", ".docx", ".pptx", ".txt")):
                try:
                    file_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                    file_data = file_obj['Body'].read()
                    file_base64 = base64.b64encode(file_data).decode("utf-8")
                    
                    # Generate presigned URL for direct download
                    try:
                        presigned_url = s3.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': S3_BUCKET, 'Key': key},
                            ExpiresIn=3600  # URL expires in 1 hour
                        )
                        original_file_url = presigned_url
                    except Exception as url_error:
                        print(f"Failed to generate presigned URL: {url_error}")
                        original_file_url = None
                    
                    original_file = {
                        "key": key,
                        "base64": file_base64,
                        "filename": Path(key).name,
                        "content_type": file_obj.get('ContentType', 'application/octet-stream'),
                        "size": len(file_data),
                        "download_url": original_file_url
                    }
                    break  # Found the original file, no need to continue
                except Exception as e:
                    original_file = {
                        "key": key,
                        "error": f"Failed to load original file: {str(e)}"
                    }
                    break

        if not original_file:
            raise HTTPException(status_code=404, detail=f"No original file found for document ID: {doc_id}")

        return {
            "doc_id": doc_id,
            "s3_prefix": prefix,
            "original_file": original_file,
            "message": "Original file retrieved successfully"
        }

    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 access failed: {e.response['Error']['Message']}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/get_source_document_url/{doc_id}")
def get_source_document_url(doc_id: str, org: str, dept: str, base_output_dir: str = "document-upload2/test-output"):
    """Get download URL, metadata, extracted text content, and images for a document."""
    prefix = f"{base_output_dir}/{org}/{dept}/{doc_id}/"

    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        if "Contents" not in response:
            raise HTTPException(status_code=404, detail=f"No document found in S3 at prefix: {prefix}")

        original_file_key = None
        original_file_info = None
        text_file_key = None
        image_keys = []

        for obj in response["Contents"]:
            key = obj["Key"]

            # Original file
            if "/original_file/" in key and key.lower().endswith((".pdf", ".docx", ".pptx", ".txt")):
                original_file_key = key
                original_file_info = {
                    "filename": Path(key).name,
                    "size": obj.get('Size', 0),
                    "last_modified": obj.get('LastModified').isoformat() if obj.get('LastModified') else None
                }

            # First extracted text
            if "/text/" in key and key.endswith(".txt") and not text_file_key:
                text_file_key = key

            # All images
            if "/images/" in key and key.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_keys.append(key)

        if not original_file_key:
            raise HTTPException(status_code=404, detail=f"No original file found for document ID: {doc_id}")

        # Generate download URL for original file
        try:
            presigned_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': original_file_key},
                ExpiresIn=3600
            )
        except Exception as url_error:
            raise HTTPException(status_code=500, detail=f"Failed to generate download URL: {str(url_error)}")

        # Get content type
        content_type, _ = mimetypes.guess_type(original_file_info["filename"])
        if not content_type:
            content_type = "application/octet-stream"

        # Read text content
        content_text = None
        if text_file_key:
            try:
                s3_obj = s3.get_object(Bucket=S3_BUCKET, Key=text_file_key)
                content_text = s3_obj['Body'].read().decode("utf-8")
            except Exception:
                content_text = None

        # Generate presigned URLs for images
        image_urls = []
        for img_key in image_keys:
            try:
                url = s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': S3_BUCKET, 'Key': img_key},
                    ExpiresIn=3600
                )
                image_urls.append({
                    "filename": Path(img_key).name,
                    "url": url
                })
            except Exception as e:
                continue  # skip if any image fails

        return {
            "doc_id": doc_id,
            "original_file": {
                "filename": original_file_info["filename"],
                "size": original_file_info["size"],
                "last_modified": original_file_info["last_modified"],
                "content_type": content_type,
                "download_url": presigned_url
            },
            "content": content_text,
            "images": image_urls,  # 🆕 Added images
            "expires_in": 3600,
            "message": "Document metadata, content, and images loaded successfully"
        }

    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 access failed: {e.response['Error']['Message']}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/metadata_status")
def get_metadata_status():
    """Get metadata database status"""
    return {
        "total_entries": len(metadata_db),
        "entries": list(metadata_db.keys())
    }

@app.post("/clear_metadata")
def clear_metadata():
    """Clear all metadata (use with caution)"""
    global metadata_db
    metadata_db.clear()
    save_metadata(metadata_db)
    return {"message": "Metadata cleared successfully"}

@app.delete("/delete_document")
def delete_document(org: str, dept: str, doc_id: str, base_output_dir: str = "document-upload2/test-output"):
    """Delete a document and all its associated files from S3, rebuild index, and update metadata"""
    global metadata_db

    try:
        prefix = f"{base_output_dir}/{org}/{dept}/{doc_id}/"

        # List all objects with this prefix
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

        if 'Contents' not in response:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        # Delete all objects with this prefix
        objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]

        if objects_to_delete:
            s3.delete_objects(
                Bucket=S3_BUCKET,
                Delete={'Objects': objects_to_delete}
            )
            print(f"[Delete] Deleted {len(objects_to_delete)} objects for document {doc_id}")

        # 🔥 Delete corresponding metadata entry
        meta_key = f"{org}/{dept}/{doc_id}"
        if meta_key in metadata_db:
            del metadata_db[meta_key]
            save_metadata(metadata_db)
            print(f"[Metadata] Removed metadata for {meta_key}")

        # 🔁 Rebuild index and metadata for remaining documents
        org_dept_prefix = f"{base_output_dir}/{org}/{dept}"
        reload_index_s3(org_dept_prefix)

        # 🧹 Purge any stale metadata after rebuild
        purge_stale_metadata()

        return {
            "status": "success",
            "message": f"Document {doc_id} and all associated files deleted successfully",
            "deleted_objects": len(objects_to_delete),
            "index_rebuilt": True,
            "metadata_updated": True
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        else:
            raise HTTPException(status_code=500, detail=f"AWS S3 error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.get("/list_documents")
def list_documents(org: str, dept: str, user_email: Optional[str] = None, user_role: Optional[str] = None, base_output_dir: str = "document-upload2/test-output"):
    """List documents for a specific organization and department with role-based filtering"""
    try:
        prefix = f"{base_output_dir}/{org}/{dept}/"
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, Delimiter='/')
        
        documents = []
        if 'CommonPrefixes' in response:
            for prefix_info in response['CommonPrefixes']:
                doc_path = prefix_info['Prefix']
                doc_id = doc_path.rstrip('/').split('/')[-1]
                
                # Get document details
                doc_prefix = f"{prefix}{doc_id}/"
                doc_response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=doc_prefix)
                
                if 'Contents' in doc_response:
                    text_files = [obj['Key'] for obj in doc_response['Contents'] if obj['Key'].endswith('.txt')]
                    image_files = [obj['Key'] for obj in doc_response['Contents'] if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    # Get ownership information
                    ownership_info = None
                    ownership_key = f"{doc_prefix}ownership.json"
                    try:
                        ownership_obj = s3.get_object(Bucket=S3_BUCKET, Key=ownership_key)
                        ownership_data = json.loads(ownership_obj['Body'].read().decode("utf-8"))
                        ownership_info = ownership_data.get("uploaded_by")
                    except:
                        # No ownership file found, document was uploaded before ownership tracking
                        pass
                    
                    # Role-based filtering
                    should_include = True
                    if user_role == "doc_owner" and user_email:
                        # doc_owner can only see their own documents
                        should_include = ownership_info == user_email
                    elif user_role == "RAG_admin":
                        # RAG_admin can see all documents in their org/dept
                        should_include = True
                    elif user_role == "RAG_user":
                        # RAG_user can see all documents (for querying)
                        should_include = True
                    
                    if should_include:
                        documents.append({
                            "doc_id": doc_id,
                            "text_chunks": len(text_files),
                            "images": len(image_files),
                            "upload_date": doc_response['Contents'][0]['LastModified'].isoformat() if doc_response['Contents'] else None,
                            "uploaded_by": ownership_info
                        })
        
        return {
            "documents": documents,
            "total_documents": len(documents),
            "user_role": user_role,
            "filtered_by_ownership": user_role == "doc_owner"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")
