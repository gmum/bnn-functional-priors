#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <notebook_path> [args...]"
    exit 1
fi

# Extract the notebook path and remaining arguments
NOTEBOOK_PATH=$1
shift  # Shift the arguments so that $@ contains the remaining arguments

# Check if the provided notebook exists
if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "Error: File '$NOTEBOOK_PATH' not found."
    exit 1
fi

# Define the output Python script path
PYTHON_SCRIPT="${NOTEBOOK_PATH%.ipynb}.py"
echo "Converting $NOTEBOOK_PATH notebook to Python script: $PYTHON_SCRIPT"

# Check if the Python script already exists
if [ -f "$PYTHON_SCRIPT" ]; then
    echo "The Python script '$PYTHON_SCRIPT' already exists. Skipping conversion."
    sleep 1
else
    # Convert the notebook to a Python script
    jupyter nbconvert --to script "$NOTEBOOK_PATH" --stdout | \
        grep -vE "get_ipython\(\)\.run_line_magic" | sed 's/os\.chdir("..")/sys.path.append("..")/g' > "$PYTHON_SCRIPT"

    # Check if the conversion was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to convert notebook to Python script."
        exit 1
    fi
fi

# Get the directory of the notebook and move to it
NOTEBOOK_DIR=$(dirname "$NOTEBOOK_PATH")
echo "Working directory: $NOTEBOOK_DIR"

# Only change directory if NOTEBOOK_DIR is not empty or "."
if [ "$NOTEBOOK_DIR" != "" ] && [ "$NOTEBOOK_DIR" != "." ]; then
    cd "$NOTEBOOK_DIR" || { echo "Error: Failed to change to directory $NOTEBOOK_DIR"; exit 1; }
    # Use the basename of the script in this case
    PYTHON_SCRIPT_FILE=$(basename "$PYTHON_SCRIPT")
else
    # If we're already in the correct directory, use the full path
    PYTHON_SCRIPT_FILE="$PYTHON_SCRIPT"
fi

# Run the converted Python script with the remaining arguments
PYTHON_SCRIPT_FILE=$(basename "$PYTHON_SCRIPT")
CURR_DIR=`pwd`
echo "EXECUTING: dir=$CURR_DIR script=$PYTHON_SCRIPT_FILE args=$@"
python "$PYTHON_SCRIPT_FILE" "$@"
echo "EXECUTION FINISHED"

# Clean up the temporary Python script
# rm "$PYTHON_SCRIPT"
exit 0
