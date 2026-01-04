#!/bin/bash
# Local testing script for macOS
# Tests everything that doesn't require GPU before deploying to RunPod

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

function print_header() {
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}========================================${NC}\n"
}

function print_test() {
    echo -e "Testing: $1"
}

function pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1\n"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

function fail() {
    echo -e "${RED}❌ FAIL${NC}: $1\n"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

function check_prereq() {
    if command -v $1 &> /dev/null; then
        pass "$1 is installed"
        return 0
    else
        fail "$1 is NOT installed"
        return 1
    fi
}

# ============================================
# Test 1: Prerequisites
# ============================================
print_header "Test 1: Prerequisites"

print_test "Docker"
check_prereq docker

print_test "Docker Compose"
check_prereq docker-compose || check_prereq "docker compose"

print_test "Python3"
check_prereq python3

print_test "Git"
check_prereq git

# ============================================
# Test 2: Config File Validation
# ============================================
print_header "Test 2: Config File Validation"

print_test "docker-compose.yml"
if docker compose config > /dev/null 2>&1; then
    pass "docker-compose.yml is valid"
else
    fail "docker-compose.yml has syntax errors"
fi

print_test "configs/train_dit_refine.yaml"
if python3 -c "import yaml; yaml.safe_load(open('configs/train_dit_refine.yaml'))" 2>/dev/null; then
    pass "train_dit_refine.yaml is valid"
else
    fail "train_dit_refine.yaml has syntax errors"
fi

print_test "configs/infer_dit_refine.yaml"
if python3 -c "import yaml; yaml.safe_load(open('configs/infer_dit_refine.yaml'))" 2>/dev/null; then
    pass "infer_dit_refine.yaml is valid"
else
    fail "infer_dit_refine.yaml has syntax errors"
fi

print_test "configs/train_vae_refine.yaml"
if python3 -c "import yaml; yaml.safe_load(open('configs/train_vae_refine.yaml'))" 2>/dev/null; then
    pass "train_vae_refine.yaml is valid"
else
    fail "train_vae_refine.yaml has syntax errors"
fi

print_test "configs/deepspeed_zero2.json"
if python3 -c "import json; json.load(open('configs/deepspeed_zero2.json'))" 2>/dev/null; then
    pass "deepspeed_zero2.json is valid"
else
    fail "deepspeed_zero2.json has syntax errors"
fi

print_test ".env.example"
if [ -f ".env.example" ]; then
    pass ".env.example exists"
else
    fail ".env.example is missing"
fi

# ============================================
# Test 3: Python Scripts Validation
# ============================================
print_header "Test 3: Python Scripts Validation"

print_test "scripts/process_dataset.py"
if python3 -m py_compile scripts/process_dataset.py 2>/dev/null; then
    pass "process_dataset.py syntax OK"
else
    fail "process_dataset.py has syntax errors"
fi

print_test "scripts/watertight_mesh.py"
if python3 -m py_compile scripts/watertight_mesh.py 2>/dev/null; then
    pass "watertight_mesh.py syntax OK"
else
    fail "watertight_mesh.py has syntax errors"
fi

print_test "scripts/blender_render.py"
if python3 -m py_compile scripts/blender_render.py 2>/dev/null; then
    pass "blender_render.py syntax OK"
else
    fail "blender_render.py has syntax errors"
fi

print_test "scripts/sample_dataset.py"
if python3 -m py_compile scripts/sample_dataset.py 2>/dev/null; then
    pass "sample_dataset.py syntax OK"
else
    fail "sample_dataset.py has syntax errors"
fi

print_test "scripts/download_pretrained.py"
if python3 -m py_compile scripts/download_pretrained.py 2>/dev/null; then
    pass "download_pretrained.py syntax OK"
else
    fail "download_pretrained.py has syntax errors"
fi

print_test "scripts/api_server.py"
if python3 -m py_compile scripts/api_server.py 2>/dev/null; then
    pass "api_server.py syntax OK"
else
    fail "api_server.py has syntax errors"
fi

# ============================================
# Test 4: Bash Scripts Validation
# ============================================
print_header "Test 4: Bash Scripts Validation"

print_test "scripts/runpod_setup.sh"
if bash -n scripts/runpod_setup.sh 2>/dev/null; then
    pass "runpod_setup.sh syntax OK"
else
    fail "runpod_setup.sh has syntax errors"
fi

print_test "scripts/runpod_monitor.sh"
if bash -n scripts/runpod_monitor.sh 2>/dev/null; then
    pass "runpod_monitor.sh syntax OK"
else
    fail "runpod_monitor.sh has syntax errors"
fi

print_test "scripts/train_deepspeed.sh"
if bash -n scripts/train_deepspeed.sh 2>/dev/null; then
    pass "train_deepspeed.sh syntax OK"
else
    fail "train_deepspeed.sh has syntax errors"
fi

print_test "train.sh"
if bash -n train.sh 2>/dev/null; then
    pass "train.sh syntax OK"
else
    fail "train.sh has syntax errors"
fi

# ============================================
# Test 5: Docker Build Test (Processing Only)
# ============================================
print_header "Test 5: Docker Build Test"

echo -e "${YELLOW}Note: Building Docker images takes 10-20 minutes.${NC}"
echo -e "${YELLOW}Set SKIP_BUILD=1 to skip this test.${NC}\n"

if [ "${SKIP_BUILD}" != "1" ]; then
    print_test "Processing container build"
    if docker compose build processing 2>&1 | tail -n 1 | grep -q "Successfully"; then
        pass "Processing container built successfully"
    else
        echo -e "${YELLOW}⚠️  WARN${NC}: Processing container build may have issues (check logs)\n"
    fi
else
    echo -e "${YELLOW}⏭️  SKIPPED${NC}: Docker builds (set SKIP_BUILD=0 to enable)\n"
fi

# ============================================
# Test 6: Directory Structure
# ============================================
print_header "Test 6: Directory Structure"

print_test "Required directories exist"
REQUIRED_DIRS=(
    "docker"
    "scripts"
    "configs"
    "docs"
)

ALL_DIRS_EXIST=true
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/"
    else
        echo "  ✗ $dir/ MISSING"
        ALL_DIRS_EXIST=false
    fi
done

if [ "$ALL_DIRS_EXIST" = true ]; then
    pass "All required directories exist"
else
    fail "Some directories are missing"
fi

print_test "Required files exist"
REQUIRED_FILES=(
    "docker/Dockerfile.processing"
    "docker/Dockerfile.training"
    "docker/Dockerfile.inference"
    "docker-compose.yml"
    "train.sh"
    "README.md"
)

ALL_FILES_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file MISSING"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = true ]; then
    pass "All required files exist"
else
    fail "Some files are missing"
fi

# ============================================
# Test 7: Git Repository
# ============================================
print_header "Test 7: Git Repository"

print_test "Git repository initialized"
if [ -d ".git" ]; then
    pass "Git repository exists"
else
    fail "Not a git repository"
fi

print_test "No uncommitted changes"
if git diff --quiet && git diff --cached --quiet; then
    pass "Working directory is clean"
else
    echo -e "${YELLOW}⚠️  WARN${NC}: You have uncommitted changes\n"
    git status --short
    echo ""
fi

# ============================================
# Summary
# ============================================
print_header "Test Summary"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
PASS_RATE=$((TESTS_PASSED * 100 / TOTAL_TESTS))

echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
else
    echo "Failed: $TESTS_FAILED"
fi
echo "Pass Rate: ${PASS_RATE}%"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # Auto-create data directories
    mkdir -p data/input data/output
    echo "✓ Created data/input/ and data/output/"
    echo ""

    echo "You're ready to test with sample data!"
    echo ""
    echo "Next steps:"
    echo "1. Place 3-5 test models in data/input/ (any format: OBJ, STL, FBX, PLY)"
    echo "2. Run: docker compose --profile processing run processing --input-dir /input --output-dir /output --limit 3"
    echo "3. Verify output with: python3 validate_data.py"
    echo "4. Push to git and deploy to RunPod"
    echo ""
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Please fix the failing tests before proceeding."
    echo ""
    exit 1
fi
