import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class StudentsDBManager:
    """
    Database manager for students information
    Handles SQLite database operations for student records
    """
    
    def __init__(self, db_path: str = "students.db"):
        """Initialize database manager with database path"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database and create tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Read and execute the SQL schema
                schema_path = os.path.join(os.path.dirname(__file__), "students_database.sql")
                if os.path.exists(schema_path):
                    with open(schema_path, 'r') as f:
                        schema = f.read()
                    cursor.executescript(schema)
                else:
                    # Fallback: create table directly
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS students (
                            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            first_name VARCHAR(50) NOT NULL,
                            last_name VARCHAR(50) NOT NULL,
                            speaks_english INTEGER CHECK (speaks_english IN (0, 1)),
                            speaks_spanish INTEGER CHECK (speaks_spanish IN (0, 1)),
                            grade_math_pass INTEGER CHECK (grade_math_pass IN (0, 1)),
                            grade_science_pass INTEGER CHECK (grade_science_pass IN (0, 1)),
                            grade_english_pass INTEGER CHECK (grade_english_pass IN (0, 1)),
                            is_highschool INTEGER CHECK (is_highschool IN (0, 1)),
                            is_active INTEGER CHECK (is_active IN (0, 1))
                        )
                    ''')
                
                conn.commit()
                print(f"Database initialized successfully at: {self.db_path}")
                
        except sqlite3.Error as e:
            print(f"Error initializing database: {e}")
    
    def add_student(self, first_name: str, last_name: str, speaks_english: int, 
                   speaks_spanish: int, grade_math_pass: int, grade_science_pass: int,
                   grade_english_pass: int, is_highschool: int, is_active: int) -> bool:
        """Add a new student to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO students (first_name, last_name, speaks_english, speaks_spanish, 
                                        grade_math_pass, grade_science_pass, grade_english_pass, 
                                        is_highschool, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (first_name, last_name, speaks_english, speaks_spanish, grade_math_pass, 
                      grade_science_pass, grade_english_pass, is_highschool, is_active))
                conn.commit()
                print(f"Student {first_name} {last_name} added successfully")
                return True
        except sqlite3.Error as e:
            print(f"Error adding student: {e}")
            return False
    
    def get_all_students(self) -> List[Dict]:
        """Retrieve all students from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # This enables column access by name
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM students")
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"Error retrieving students: {e}")
            return []
    
    def get_student_by_id(self, student_id: int) -> Optional[Dict]:
        """Retrieve a specific student by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            print(f"Error retrieving student: {e}")
            return None
    
    def update_student(self, student_id: int, **kwargs) -> bool:
        """Update student information"""
        try:
            if not kwargs:
                return False
            
            # Build dynamic update query
            set_clause = ", ".join([f"{key} = ?" for key in kwargs.keys()])
            values = list(kwargs.values()) + [student_id]
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"UPDATE students SET {set_clause} WHERE student_id = ?", values)
                conn.commit()
                
                if cursor.rowcount > 0:
                    print(f"Student ID {student_id} updated successfully")
                    return True
                else:
                    print(f"No student found with ID {student_id}")
                    return False
                    
        except sqlite3.Error as e:
            print(f"Error updating student: {e}")
            return False
    
    def delete_student(self, student_id: int) -> bool:
        """Delete a student from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    print(f"Student ID {student_id} deleted successfully")
                    return True
                else:
                    print(f"No student found with ID {student_id}")
                    return False
                    
        except sqlite3.Error as e:
            print(f"Error deleting student: {e}")
            return False
    
    def search_students(self, search_term: str) -> List[Dict]:
        """Search students by name"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM students 
                    WHERE first_name LIKE ? OR last_name LIKE ?
                """, (f"%{search_term}%", f"%{search_term}%"))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"Error searching students: {e}")
            return []
    
    def get_student_count(self) -> int:
        """Get total number of students"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM students")
                return cursor.fetchone()[0]
        except sqlite3.Error as e:
            print(f"Error getting student count: {e}")
            return 0
    
    def get_students_by_language(self, language: str) -> List[Dict]:
        """Get students who speak a specific language"""
        try:
            column = f"speaks_{language.lower()}"
            if column not in ['speaks_english', 'speaks_spanish']:
                return []
                
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM students WHERE {column} = 1")
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"Error retrieving students by language: {e}")
            return []
    
    def get_students_by_school_level(self, is_highschool: bool) -> List[Dict]:
        """Get students by school level (True for high school, False for primary)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM students WHERE is_highschool = ?", (1 if is_highschool else 0,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"Error retrieving students by school level: {e}")
            return []
    
    def get_failing_students(self) -> List[Dict]:
        """Get students who failed at least one subject"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM students 
                    WHERE grade_math_pass = 0 OR grade_science_pass = 0 OR grade_english_pass = 0
                """)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"Error retrieving failing students: {e}")
            return []

    def close(self):
        """Close database connection (if needed for connection pooling)"""
        pass  # SQLite connections are handled per transaction


def main():
    """Example usage of the StudentsDBManager"""
    db_manager = StudentsDBManager()
    
    # Example: Add some sample students
    # Parameters: first_name, last_name, speaks_english, speaks_spanish, grade_math_pass, grade_science_pass, grade_english_pass, is_highschool, is_active
    sample_students = [
        ("Alice", "Johnson", 1, 0, 1, 1, 1, 1, 1),  # Speaks English, passed all subjects, highschool, active
        ("Bob", "Smith", 1, 1, 0, 1, 1, 0, 1),      # Speaks both languages, failed math, primary school, active
        ("Charlie", "Brown", 0, 1, 1, 0, 1, 1, 1),  # Speaks Spanish, failed science, highschool, active
        ("Maria", "Garcia", 0, 1, 1, 1, 0, 0, 1),   # Speaks Spanish, failed English, primary school, active
        ("John", "Doe", 1, 0, 1, 1, 1, 1, 0)        # Speaks English, passed all subjects, highschool, inactive
    ]
    
    print("Adding sample students...")
    for student in sample_students:
        db_manager.add_student(*student)
    
    print(f"\nTotal students in database: {db_manager.get_student_count()}")
    
    print("\nAll students:")
    students = db_manager.get_all_students()
    for student in students:
        school_level = "High School" if student['is_highschool'] else "Primary School"
        status = "Active" if student['is_active'] else "Inactive"
        languages = []
        if student['speaks_english']: languages.append("English")
        if student['speaks_spanish']: languages.append("Spanish")
        lang_str = ", ".join(languages) if languages else "None"
        
        print(f"ID: {student['student_id']}, Name: {student['first_name']} {student['last_name']}")
        print(f"   School: {school_level}, Status: {status}, Languages: {lang_str}")
        print(f"   Grades - Math: {'Pass' if student['grade_math_pass'] else 'Fail'}, "
              f"Science: {'Pass' if student['grade_science_pass'] else 'Fail'}, "
              f"English: {'Pass' if student['grade_english_pass'] else 'Fail'}")


if __name__ == "__main__":
    main() 
    