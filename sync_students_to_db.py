import os
import re
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATASET_PATH = os.path.join(os.getcwd(), 'public')

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def sync_students_to_database():
    """
    Scans the 'public' folder for student folders and syncs them to Supabase.
    Folder format: 'RollNo_StudentName' (e.g., '07_Samiksha_Pawar')
    """
    students_to_insert = []
    
    print(f"Scanning directory: {DATASET_PATH}")
    
    # Iterate through all student folders
    for folder_name in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        try:
            # Extract RollNo and Name from folder name
            match = re.match(r'(\d+)_(.+)', folder_name)
            if not match:
                print(f"Warning: Skipping folder '{folder_name}'. Format should be 'RollNo_Name'.")
                continue
            
            roll_no = int(match.group(1))
            name = match.group(2).replace('_', ' ')  # Replace underscores with spaces
            
            students_to_insert.append({
                'roll_no': roll_no,
                'name': name
            })
            
            print(f"Found student: Roll {roll_no} - {name}")
            
        except Exception as e:
            print(f"Error processing folder '{folder_name}': {e}")
    
    # Insert students into Supabase (upsert to avoid duplicates)
    if students_to_insert:
        try:
            # First, check if students already exist
            existing_response = supabase.table('students').select('roll_no').execute()
            existing_roll_nos = {student['roll_no'] for student in existing_response.data}
            
            # Filter out students that already exist
            new_students = [s for s in students_to_insert if s['roll_no'] not in existing_roll_nos]
            
            if new_students:
                response = supabase.table('students').insert(new_students).execute()
                print(f"\n✅ Successfully added {len(new_students)} new students to the database!")
            else:
                print("\n✅ All students already exist in the database!")
            
            # Display all students in database
            all_students = supabase.table('students').select('*').execute()
            print(f"\n📊 Total students in database: {len(all_students.data)}")
            print("\nStudent List:")
            for student in sorted(all_students.data, key=lambda x: x['roll_no']):
                print(f"  Roll {student['roll_no']:02d}: {student['name']}")
                
        except Exception as e:
            print(f"\n❌ Error syncing students to database: {e}")
            return False
    else:
        print("\n⚠️ No students found to sync!")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("   Student Database Sync Tool")
    print("=" * 60)
    print()
    
    success = sync_students_to_database()
    
    print()
    print("=" * 60)
    if success:
        print("✅ Sync completed successfully!")
    else:
        print("❌ Sync failed. Please check the errors above.")
    print("=" * 60)
