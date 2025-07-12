SELECT * 
FROM students 
WHERE is_active = 0 
  AND is_highschool = 1 
  AND speaks_english = 1 
  AND grade_english_pass = 1 
  AND (grade_math_pass = 0 OR grade_science_pass = 0);