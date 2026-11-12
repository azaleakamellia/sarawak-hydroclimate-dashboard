import os

# 1. Define your target folder and the output file name
folder_path = r"D:\my_project\src"
output_file = r"D:\my_project\src\compiled_code.txt"
# 2. Specify file extensions you want to include (leave empty or add more as needed)
# Set to None or empty set if you want to include ALL files
valid_extensions = {".py", ".js", ".html", ".css", ".cpp", ".c", ".java", ".txt"}

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()

            # Skip the output file itself if it's inside the same directory
            if file == output_file:
                continue

            # Filter by extension if valid_extensions is set
            if valid_extensions and file_ext not in valid_extensions:
                continue

            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, folder_path)

            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    # Write a separator header for each code file
                    outfile.write(f"\n{'='*50}\n")
                    outfile.write(f"FILE: {relative_path}\n")
                    outfile.write(f"{'='*50}\n\n")

                    # Write the code contents
                    outfile.write(infile.read())
                    outfile.write("\n\n")

            except Exception as e:
                print(f"Skipped {file_path} due to error: {e}")

print(f"All code files have been compiled into '{output_file}'!")
