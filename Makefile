# Compiler: Default to gcc, can be overridden e.g., make CC=clang
CC = gcc

# Base CFLAGS (applied to all builds)
# -std=gnu11: GNU C11 standard (supports C11 atomics and GNU extensions)
# -O3: Optimization level 3
# -Wall -Wextra: Enable most warnings
# -Iinclude: Add include directory
# -g: Include debugging symbols
# -ffast-math: Aggressive floating point optimizations (can sometimes reduce precision)
# -march=native: Optimize for the current machine's architecture (less portable binary)
BASE_CFLAGS = -std=gnu11 -O3 -Wall -Wextra -Iinclude -g -ffast-math -march=native

# Base LDFLAGS (applied to all builds)
# -lm: Link math library
BASE_LDFLAGS = -lm

# Initialize CFLAGS and LDFLAGS with base values
CFLAGS = $(BASE_CFLAGS)
LDFLAGS = $(BASE_LDFLAGS)

# Detect OS
UNAME_S := $(shell uname -s)

# Platform-specific flags
ifeq ($(UNAME_S),Darwin) # macOS
    CFLAGS += -Xpreprocessor -fopenmp # For Clang OpenMP
    LDFLAGS += -lomp                # Link libomp for Clang

    HOMEBREW_PREFIX := $(shell brew --prefix 2>/dev/null)

    # libomp from Homebrew
    LIBOMP_BREW_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
    ifneq ($(LIBOMP_BREW_PREFIX),)
        CFLAGS += -I$(LIBOMP_BREW_PREFIX)/include
        LDFLAGS += -L$(LIBOMP_BREW_PREFIX)/lib
    else ifneq ($(HOMEBREW_PREFIX),)
        # Fallback to general Homebrew path if specific libomp prefix not found
        # This might be needed if libomp is part of llvm or another package
        CFLAGS += -I$(HOMEBREW_PREFIX)/include
        LDFLAGS += -L$(HOMEBREW_PREFIX)/lib
        $(warning "libomp specific prefix not found, using general Homebrew prefix for OpenMP. Ensure libomp is installed and accessible.")
    else
        $(warning "Homebrew not found or libomp not found via Homebrew. OpenMP might not link correctly on macOS.")
    endif

    # FFTW from Homebrew
    FFTW_BREW_PREFIX := $(shell brew --prefix fftw 2>/dev/null)
    ifneq ($(FFTW_BREW_PREFIX),)
        CFLAGS += -I$(FFTW_BREW_PREFIX)/include
        LDFLAGS += -L$(FFTW_BREW_PREFIX)/lib -lfftw3
    else ifneq ($(HOMEBREW_PREFIX),)
         CFLAGS += -I$(HOMEBREW_PREFIX)/include # General Homebrew include
         LDFLAGS += -L$(HOMEBREW_PREFIX)/lib -lfftw3 # General Homebrew lib path + link fftw3
         $(warning "FFTW specific prefix not found, using general Homebrew prefix for FFTW. Ensure fftw is installed and accessible.")
    else
        $(warning "Homebrew not found or FFTW not found via Homebrew. Attempting to link FFTW from system paths on macOS.")
        LDFLAGS += -lfftw3 # Fallback if Homebrew FFTW is not found
    endif

else # Linux (and other Unix-like systems)
    CFLAGS += -fopenmp
    LDFLAGS += -fopenmp -pthread -latomic # -pthread for pthreads, -latomic for C11 atomics

    # FFTW3 using pkg-config on Linux
    PKG_CONFIG_FFTW_CFLAGS := $(shell pkg-config --cflags fftw3 2>/dev/null)
    PKG_CONFIG_FFTW_LDFLAGS := $(shell pkg-config --libs fftw3 2>/dev/null)

    ifeq ($(PKG_CONFIG_FFTW_CFLAGS)$(PKG_CONFIG_FFTW_LDFLAGS),) # Check if both are empty
        $(warning "pkg-config could not find fftw3. Ensure libfftw3-dev is installed. Attempting basic link.")
        LDFLAGS += -lfftw3 # Basic fallback
    else
        CFLAGS += $(PKG_CONFIG_FFTW_CFLAGS)
        LDFLAGS += $(PKG_CONFIG_FFTW_LDFLAGS)
    endif
endif

# Directories
SRC_DIR = src
OBJ_DIR = obj

# Source files
SRCS = $(SRC_DIR)/cli.c \
       $(SRC_DIR)/diffraction_engine.c \
       $(SRC_DIR)/globals.c \
       $(SRC_DIR)/quantum_engine.c \
       $(SRC_DIR)/render_engine.c \
       $(SRC_DIR)/smiles_parser.c \
       $(SRC_DIR)/utils.c

# Object files
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

# Target executable
EXEC = hdf_plus

.PHONY: all clean install uninstall debug profile compile_commands help test depend export-compile-cmd

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) # Use $^ for all prerequisites (OBJS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR) # Add $(OBJ_DIR) as an order-only prerequisite
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(EXEC) profile_output.txt gmon.out compile_commands.json Makefile.deps compile_commands.sh test_output

# Install to system path (respect DESTDIR for staging)
install: $(EXEC)
	install -d $(DESTDIR)/usr/local/bin/
	install -m 755 $(EXEC) $(DESTDIR)/usr/local/bin/

# Remove from system path (respect DESTDIR)
uninstall:
	rm -f $(DESTDIR)/usr/local/bin/$(EXEC)

# Debug build: Override CFLAGS for debug
debug: CFLAGS_DEBUG = $(BASE_CFLAGS) -std=gnu11 -DDEBUG -O0 -g # Reset CFLAGS for debug, keep -g and -Iinclude
debug: LDFLAGS_DEBUG = $(BASE_LDFLAGS) # Reset LDFLAGS for debug
debug:
	$(MAKE) CFLAGS="$(CFLAGS_DEBUG)" LDFLAGS="$(LDFLAGS_DEBUG)" $(EXEC) # Rebuild with new flags

# Profiling build: Append -pg to existing flags
profile: CFLAGS_PROFILE = $(CFLAGS) -pg
profile: LDFLAGS_PROFILE = $(LDFLAGS) -pg
profile: clean
	$(MAKE) CFLAGS="$(CFLAGS_PROFILE)" LDFLAGS="$(LDFLAGS_PROFILE)" $(EXEC)
	@echo "Built with profiling information. Run ./$(EXEC) and then use:"
	@echo "gprof ./$(EXEC) gmon.out > profile_output.txt"


# Generate compile_commands.json for IDE integration
compile_commands:
	@echo "[" > compile_commands.json
	@printf "Generating compile_commands.json...\\n"
	@first=true; \
	for src_file in $(SRCS); do \
		obj_file=$(OBJ_DIR)/$$(basename $$src_file .c).o; \
		if [ "$$first" = false ]; then printf ",\\n" >> compile_commands.json; fi; \
		printf "  {\n" >> compile_commands.json; \
		printf "    \"directory\": \"$(shell pwd)\",\n" >> compile_commands.json; \
		printf "    \"command\": \"$(CC) $(CFLAGS) -c %s -o %s\",\n" "$$src_file" "$$obj_file" >> compile_commands.json; \
		printf "    \"file\": \"%s\"\n" "$$src_file" >> compile_commands.json; \
		printf "  }" >> compile_commands.json; \
		first=false; \
	done
	@printf "\\n]\n" >> compile_commands.json
	@echo "Generated compile_commands.json"

# Generate dependency information
depend:
	@echo "Generating dependencies into Makefile.deps..."
	@rm -f Makefile.deps
	@for src in $(SRCS); do \
		$(CC) -MM $(CFLAGS) $$src | sed 's|^\(.*\)\.o:|$(OBJ_DIR)/\1.o:|' >> Makefile.deps; \
	done
	@echo "Dependencies generated."

# Export compile commands for troubleshooting
export-compile-cmd:
	@echo "Exporting compile commands to compile_commands.sh"
	@echo "#!/bin/bash" > compile_commands.sh
	@echo "# This file contains the commands used to build hdf_plus" >> compile_commands.sh
	@echo "# Run this script to manually compile individual files for troubleshooting" >> compile_commands.sh
	@echo "" >> compile_commands.sh
	@echo "mkdir -p $(OBJ_DIR)" >> compile_commands.sh
	@echo "" >> compile_commands.sh
	@for src in $(SRCS); do \
		echo "# Compile $$(basename $$src)" >> compile_commands.sh; \
		echo "$(CC) $(CFLAGS) -c $$src -o $(OBJ_DIR)/$$(basename $$src .c).o" >> compile_commands.sh; \
		echo "" >> compile_commands.sh; \
	done
	@echo "# Link final executable" >> compile_commands.sh
	@echo "$(CC) $(CFLAGS) $(OBJS) -o $(EXEC) $(LDFLAGS)" >> compile_commands.sh
	@chmod +x compile_commands.sh
	@echo "Generated compile_commands.sh"

# Test target
test: $(EXEC)
	@echo "Running basic tests..."
	@mkdir -p test_output
	@./$(EXEC) --help > /dev/null 2>&1 || (echo "Help command failed"; exit 1)
	@echo "Help command executed successfully"
	@if [ -f "test_data/test.csv" ]; then \
		./$(EXEC) -i test_data/test.csv -o test_output/test_out.csv --output-dir test_output/images -n || (echo "Test run failed"; exit 1); \
		echo "Test data processed successfully (images suppressed for test)"; \
	elif [ -d "test_data" ]; then \
		echo "No test_data/test.csv found. Skipping data processing test."; \
	else \
		echo "No test_data directory found. Skipping data processing test."; \
	fi
	@echo "Basic tests completed."


# Help information
help:
	@echo "HDF Plus Makefile Help"
	@echo "======================="
	@echo "Available targets:"
	@echo "  all               : Build the main executable (default)"
	@echo "  clean             : Remove all build files and test outputs"
	@echo "  install           : Install the executable to /usr/local/bin (use with sudo)"
	@echo "  uninstall         : Remove the executable from /usr/local/bin (use with sudo)"
	@echo "  debug             : Build with debug flags (-O0 -g -DDEBUG)"
	@echo "  profile           : Build with profiling information (-pg)"
	@echo "  compile_commands  : Generate compile_commands.json for IDEs"
	@echo "  depend            : Generate Makefile.deps for dependency tracking"
	@echo "  export-compile-cmd: Export compilation commands to a shell script"
	@echo "  test              : Run basic tests (checks --help and processes test_data/test.csv if available)"
	@echo "  help              : Display this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make              : Build the project"
	@echo "  make clean        : Clean build files"
	@echo "  make install      : sudo make install"
	@echo "  make profile      : Build for profiling"
	@echo "  make CC=clang     : Build using clang compiler"
	@echo ""
	@echo "Common Dependencies:"
	@echo "  - C Compiler (gcc or clang)"
	@echo "  - Make utility"
	@echo "  - FFTW3 library (Ubuntu: libfftw3-dev, macOS: brew install fftw)"
	@echo "  - OpenMP (Usually part of compiler toolchain; macOS may need: brew install libomp)"
	@echo "  - (Linux) libpthread, libatomic (usually part of glibc/toolchain)"
	@echo ""
	@echo "Environment configuration (can be overridden):"
	@echo "  - CC=$(CC)"
	@echo "  - Current CFLAGS=$(CFLAGS)"
	@echo "  - Current LDFLAGS=$(LDFLAGS)"

# Include dependency information if it exists. Generate with 'make depend'.
-include Makefile.deps