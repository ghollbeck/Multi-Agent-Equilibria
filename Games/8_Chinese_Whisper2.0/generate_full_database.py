#!/usr/bin/env python3
"""
generate_full_database.py - Generate a comprehensive database with all possible student parameter combinations
"""
import sqlite3
import itertools
from typing import List, Tuple
import os

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_success(message: str):
    print(f"{Colors.GREEN}✅{Colors.ENDC} {message}")

def print_error(message: str):
    print(f"{Colors.RED}❌{Colors.ENDC} {message}")

def print_info(message: str):
    print(f"{Colors.BLUE}ℹ️{Colors.ENDC} {message}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠️{Colors.ENDC} {message}")


class FullDatabaseGenerator:
    """Generate a comprehensive database with all possible student combinations"""
    
    def __init__(self, db_path: str = "full_students.db"):
        self.db_path = db_path
        self.binary_params = [
            'speaks_english',
            'speaks_spanish', 
            'grade_math_pass',
            'grade_science_pass',
            'grade_english_pass',
            'is_highschool',
            'is_active'
        ]
        
    def generate_all_combinations(self) -> List[Tuple]:
        """Generate all possible combinations of binary parameters"""
        print_info(f"Generating all combinations for {len(self.binary_params)} binary parameters...")
        
        # Generate all combinations of 0s and 1s for the binary parameters
        combinations = list(itertools.product([0, 1], repeat=len(self.binary_params)))
        
        print_success(f"Generated {len(combinations)} unique parameter combinations (2^{len(self.binary_params)} = {2**len(self.binary_params)})")
        return combinations
    
    def create_database(self):
        """Create the database with all possible student combinations"""
        print_info(f"Creating database at: {self.db_path}")
        
        # Remove existing database if it exists
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            print_info("Removed existing database file")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create the students table
                print_info("Creating students table schema...")
                cursor.execute('''
                    CREATE TABLE students (
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
                print_success("Students table created successfully")
                
                # Generate all combinations
                combinations = self.generate_all_combinations()
                
                # Insert all students
                print_info("Inserting students into database...")
                students_data = []
                
                for i, combination in enumerate(combinations, 1):
                    first_name = f"Alice{i}"
                    last_name = "Student"
                    
                    # Unpack the combination tuple for the binary parameters
                    student_tuple = (first_name, last_name) + combination
                    students_data.append(student_tuple)
                
                # Batch insert for efficiency
                cursor.executemany('''
                    INSERT INTO students (first_name, last_name, speaks_english, speaks_spanish, 
                                        grade_math_pass, grade_science_pass, grade_english_pass, 
                                        is_highschool, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', students_data)
                
                conn.commit()
                print_success(f"Successfully inserted {len(students_data)} students into the database")
                
                # Verify the data
                cursor.execute("SELECT COUNT(*) FROM students")
                count = cursor.fetchone()[0]
                print_success(f"Database verification: {count} students in total")
                
                # Show some sample data
                print_info("Sample data preview:")
                cursor.execute("SELECT * FROM students LIMIT 5")
                samples = cursor.fetchall()
                
                print_info("First 5 students:")
                for student in samples:
                    print_info(f"  ID: {student[0]}, Name: {student[1]} {student[2]}, Parameters: {student[3:]}")
                
                print_info("Last 5 students:")
                cursor.execute("SELECT * FROM students ORDER BY student_id DESC LIMIT 5")
                samples = cursor.fetchall()
                for student in samples:
                    print_info(f"  ID: {student[0]}, Name: {student[1]} {student[2]}, Parameters: {student[3:]}")
                    
        except Exception as e:
            print_error(f"Failed to create database: {e}")
            raise
    
    def verify_completeness(self):
        """Verify that all possible combinations are present"""
        print_info("Verifying database completeness...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check total count
                cursor.execute("SELECT COUNT(*) FROM students")
                total_count = cursor.fetchone()[0]
                expected_count = 2 ** len(self.binary_params)
                
                if total_count == expected_count:
                    print_success(f"✓ Total count correct: {total_count} students")
                else:
                    print_error(f"✗ Total count mismatch: {total_count} found, {expected_count} expected")
                
                # Check for unique combinations
                param_columns = ', '.join(self.binary_params)
                cursor.execute(f"SELECT DISTINCT {param_columns} FROM students")
                unique_combinations = cursor.fetchall()
                
                if len(unique_combinations) == expected_count:
                    print_success(f"✓ All combinations unique: {len(unique_combinations)} distinct parameter sets")
                else:
                    print_error(f"✗ Combination uniqueness issue: {len(unique_combinations)} unique, {expected_count} expected")
                
                # Check for any missing combinations
                all_expected = self.generate_all_combinations()
                cursor.execute(f"SELECT {param_columns} FROM students ORDER BY student_id")
                all_actual = cursor.fetchall()
                
                missing = set(all_expected) - set(all_actual)
                if not missing:
                    print_success("✓ No missing combinations found")
                else:
                    print_error(f"✗ {len(missing)} missing combinations found")
                    for combo in list(missing)[:5]:  # Show first 5 missing
                        print_error(f"  Missing: {combo}")
                
        except Exception as e:
            print_error(f"Verification failed: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive students database")
    parser.add_argument("--output", default="full_students.db", help="Output database file path")
    parser.add_argument("--verify", action="store_true", help="Verify database completeness after creation")
    args = parser.parse_args()
    
    print_info("=" * 60)
    print_info("Full Students Database Generator")
    print_info("=" * 60)
    
    # Determine script directory for output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, args.output)
    
    try:
        generator = FullDatabaseGenerator(db_path)
        generator.create_database()
        
        if args.verify:
            generator.verify_completeness()
        
        print_info("=" * 60)
        print_success("Database generation completed successfully!")
        print_info(f"Database saved to: {db_path}")
        print_info("To use this database in evaluate_sql.py, update the script to load from this file.")
        
    except Exception as e:
        print_error(f"Database generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 