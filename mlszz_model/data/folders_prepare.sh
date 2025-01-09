
#!/bin/bash

# Define a list of project names
projects=("cpython" "bitcoin" "spring-boot" "spring-framework")

# Base directory where folders will be created
base_dir="$(cd "$(dirname "$0")" && pwd)"


# Iterate through the list of project names
for project in "${projects[@]}"; do
  project_path="$base_dir/$project"

  if [ -d "$project_path" ]; then
    echo "Folder '$project_path' already exists."
  else
    echo "Creating folder '$project_path'..."
    mkdir "$project_path"
  fi

  # Define the required file names
  file1="${project}_bszz_inducing_commits_features.csv"
  file2="${project}_git_features.csv"

  # Check if the required files exist and show warnings if not
  if [ ! -f "$project_path/$file1" ]; then
    echo "Warning: File '$file1' does not exist in '$project_path'."
  fi

  if [ ! -f "$project_path/$file2" ]; then
    echo "Warning: File '$file2' does not exist in '$project_path'."
  fi

done

echo "All project folders checked or created."