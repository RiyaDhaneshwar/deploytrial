import pandas as pd
import os

def import_csv(csv_path):
    if not os.path.exists(csv_path):
        # Create new CSV with default structure if it doesn't exist
        df = pd.DataFrame({'name': [], 'organization': [], 'role': [], 'email': []})
        save_csv(df, csv_path)
        return df
    
    df = pd.read_csv(csv_path)
    
    # Check if required columns exist
    if 'email' not in df.columns or 'role' not in df.columns:
        # Create new CSV with correct structure
        df = pd.DataFrame({'name': [], 'organization': [], 'role': [], 'email': []})
        save_csv(df, csv_path)
    
    return df


def add_row(df, row_dict):
    return pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)

def save_csv(df, csv_path):
    # Only create directory if path has a directory component
    if os.path.dirname(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)


def email_exists(df, email):
    return email in df['email'].values


def get_row_index_by_email(df, email):
    matches = df.index[df['email'] == email].tolist()
    return matches[0] if matches else -1


def get_user_role(df, email):
    """Get role for a specific email"""
    if email_exists(df, email):
        user_row = df[df['email'] == email]
        return user_row['role'].iloc[0]
    return None


def get_user_organization(df, email):
    """Get organization for a specific email"""
    if email_exists(df, email):
        user_row = df[df['email'] == email]
        return user_row['organization'].iloc[0]
    return None


def update_row_by_email(df, email, new_role):
    idx = get_row_index_by_email(df, email)
    if idx == -1:
        return df, False
    df.at[idx, 'role'] = new_role
    return df, True


def delete_row_by_email(df, email):
    idx = get_row_index_by_email(df, email)
    if idx == -1:
        return df, False
    df = df.drop(index=idx).reset_index(drop=True)
    return df, True
