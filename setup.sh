#!/bin/bash

# NYC Taxi ML Project - Setup Script
# This script sets up the environment for the project

set -e  # Exit on error

echo "=========================================="
echo "NYC Taxi ML Project - Environment Setup"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo "Please do not run this script as root"
    exit 1
fi

# 1. Check and install Java
echo "Step 1: Checking Java installation..."
if ! command -v java &> /dev/null; then
    echo "Java not found. Installing OpenJDK 11..."
    sudo apt update
    sudo apt install -y openjdk-11-jdk-headless
else
    echo "✓ Java is already installed"
    java -version
fi

# 2. Set JAVA_HOME
echo ""
echo "Step 2: Setting JAVA_HOME..."
JAVA_PATH=$(update-alternatives --query java | grep 'Value:' | awk '{print $2}')
JAVA_HOME_PATH=$(dirname $(dirname $JAVA_PATH))

if [ -z "$JAVA_HOME" ]; then
    echo "export JAVA_HOME=$JAVA_HOME_PATH" >> ~/.bashrc
    echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ~/.bashrc
    export JAVA_HOME=$JAVA_HOME_PATH
    export PATH=$JAVA_HOME/bin:$PATH
    echo "✓ JAVA_HOME set to: $JAVA_HOME"
else
    echo "✓ JAVA_HOME already set to: $JAVA_HOME"
fi

# 3. Create virtual environment
echo ""
echo "Step 3: Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# 4. Activate and install dependencies
echo ""
echo "Step 4: Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# 5. Create necessary directories
echo ""
echo "Step 5: Creating project directories..."
mkdir -p data/raw data/bronze data/silver data/gold
mkdir -p models results tableau docs/spark_ui_screenshots
mkdir -p docs/spark-events
echo "✓ Directories created"

# 6. Verify PySpark installation
echo ""
echo "Step 6: Verifying PySpark installation..."
python -c "from pyspark.sql import SparkSession; print('✓ PySpark is working correctly')" || {
    echo "✗ PySpark verification failed"
    exit 1
}

# 7. Create event log directory
echo ""
echo "Step 7: Setting up Spark event logging..."
mkdir -p docs/spark-events
chmod 755 docs/spark-events
echo "✓ Spark event logging configured"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start Jupyter Lab: jupyter lab"
echo "3. Open notebooks/01_data_ingestion.ipynb"
echo ""
echo "Note: You may need to restart your terminal or run:"
echo "  source ~/.bashrc"
echo "to apply JAVA_HOME changes."
echo ""
