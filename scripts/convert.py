#!/usr/bin/env python3
"""
High-Performance Image Converter: Scans a directory and converts all PPM/PGM files to JPEG or PNG format.
Shows minimal progress output with dots for each completed batch.

Usage:
    python image_converter.py -i <input_directory> -o <output_directory> -f <jpeg|png> [options]

Options:
    -i, --input_dir    Input directory containing PPM/PGM files
    -o, --output_dir   Output directory for converted files
    -f, --format       Output format: jpeg or png (default: png)
    -q, --quality      JPEG quality 1-100 (default: 85)
    -r, --recursive    Process subdirectories recursively
    -j, --processes    Number of parallel processes (default: CPU count - 1)
    -b, --batch_size   Files per batch (default: 20)
    -s, --buffer_size  Buffer size in bytes (default: 10MB)
    -k, --quick        Quick mode (faster, larger files)

Example:
    python image_converter.py -i ./hdf_results/images -o ./converted -f png -j 6 -k
"""

import os
import argparse
import sys
import time
from PIL import Image, ImageFile
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool, Queue, Value, Lock
import mmap
import io

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Global counter for progress tracking
counter = None
lock = None

def parse_arguments():
    """Parse command line arguments with optimized defaults and short options."""
    parser = argparse.ArgumentParser(description='High-Performance converter for PPM/PGM files to JPEG or PNG format.')
    parser.add_argument('-i', '--input_dir', required=True, help='Input directory containing PPM/PGM files')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory for converted files')
    parser.add_argument('-f', '--format', choices=['jpeg', 'png'], default='png', help='Output format (default: png)')
    parser.add_argument('-q', '--quality', type=int, default=85, help='Quality for JPEG compression (1-100, default: 85)')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively process subdirectories')
    parser.add_argument('-j', '--processes', type=int, default=max(os.cpu_count() - 1, 1), 
                        help=f'Number of conversion processes (default: {max(os.cpu_count() - 1, 1)})')
    parser.add_argument('-b', '--batch_size', type=int, default=20, 
                        help='Number of files to process in each batch (default: 20)')
    parser.add_argument('-s', '--buffer_size', type=int, default=10485760,  # 10MB
                        help='Buffer size for file operations in bytes (default: 10MB)')
    parser.add_argument('-k', '--quick', action='store_true', 
                        help='Quick mode: disables optimization for faster conversion but larger files')
    
    return parser.parse_args()

def init_worker(counter_val, lock_val):
    """Initialize worker process with shared counter and lock."""
    global counter, lock
    counter = counter_val
    lock = lock_val

def convert_image_optimized(args):
    """Convert a single image with optimized settings."""
    input_path, output_path, output_format, quality, quick_mode, buffer_size = args
    
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Memory mapping for efficient file reading
        with open(input_path, 'rb') as f_in:
            # For very large files, use mmap
            if os.path.getsize(input_path) > buffer_size:
                with mmap.mmap(f_in.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    img = Image.open(io.BytesIO(mm))
                    img.load()  # Load image data before closing the file
            else:
                # For smaller files, read directly
                img = Image.open(f_in)
                img.load()
        
        # Optimize processing based on image type and output format
        if img.mode in ('RGBA', 'LA') and output_format.lower() == 'jpeg':
            # JPEG doesn't support alpha, convert to RGB
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else img.split()[1])
            img = background
        
        # Prepare save parameters
        save_params = {}
        if output_format.lower() == 'jpeg':
            save_params['quality'] = quality
            save_params['optimize'] = not quick_mode
            save_params['progressive'] = not quick_mode
        elif output_format.lower() == 'png':
            save_params['optimize'] = not quick_mode
            save_params['compress_level'] = 1 if quick_mode else 6
        
        # Use a memory buffer for the conversion to minimize disk I/O
        with open(output_path, 'wb') as f_out:
            img.save(f_out, format=output_format.upper(), **save_params)
        
        # Update progress counter
        with lock:
            counter.value += 1
        
        return True, input_path
        
    except Exception as e:
        return False, f"{input_path}: {str(e)}"
    finally:
        # Ensure image is closed to free memory
        if 'img' in locals():
            img.close()

def find_files_optimized(input_dir, recursive, extensions=('.ppm', '.pgm')):
    """Find files with optimized generator to reduce memory usage."""
    input_path = Path(input_dir)
    
    if not recursive:
        # Non-recursive mode - faster for flat directories
        for ext in extensions:
            yield from input_path.glob(f'*{ext}')
    else:
        # For recursive mode, use os.walk which is more efficient than Path.glob('**/*')
        for root, _, files in os.walk(input_path):
            root_path = Path(root)
            for file in files:
                if file.lower().endswith(extensions):
                    yield root_path / file

def process_batches(args, file_list, queue_size=20):
    """Process files in batches using a process pool with minimal progress output."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_format = args.format
    quality = args.quality
    quick_mode = args.quick
    buffer_size = args.buffer_size
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup shared counter for progress reporting
    global_counter = Value('i', 0)
    global_lock = Lock()
    
    total_files = len(file_list)
    if total_files == 0:
        print("No PPM/PGM files found.")
        return 0
    
    print(f"Converting {total_files} files...", end="", flush=True)
    
    # Start time for performance measurement
    start_time = time.time()
    
    # Create and configure the process pool
    with Pool(processes=args.processes, 
              initializer=init_worker, 
              initargs=(global_counter, global_lock)) as pool:
        
        # Prepare the conversion tasks
        conversion_tasks = []
        for input_file in file_list:
            relative_path = input_file.relative_to(input_dir)
            output_file = output_dir / relative_path.with_suffix(f".{output_format}")
            
            # Create task parameters
            task = (input_file, output_file, output_format, quality, quick_mode, buffer_size)
            conversion_tasks.append(task)
        
        # Process in batches to prevent memory issues with large directories
        batch_size = min(args.batch_size, total_files)
        batches = [conversion_tasks[i:i + batch_size] for i in range(0, len(conversion_tasks), batch_size)]
        
        # Submit all batches to the pool
        results = []
        for i, batch in enumerate(batches):
            batch_results = pool.map_async(convert_image_optimized, batch)
            results.append(batch_results)
        
        # Monitor progress and show minimal updates
        success_count = 0
        error_count = 0
        errors = []
        completed_batches = 0
        total_batches = len(batches)
        
        # Wait for all batches to complete, showing minimal progress
        while completed_batches < total_batches:
            # Check each result to see if it's complete
            for i, result in enumerate(results):
                if result.ready() and not hasattr(result, '_processed'):
                    # Mark as processed to avoid counting it again
                    setattr(result, '_processed', True)
                    completed_batches += 1
                    
                    # Print a dot for each completed batch
                    print(".", end="", flush=True)
                    
                    # Record results
                    for success, msg in result.get():
                        if success:
                            success_count += 1
                        else:
                            error_count += 1
                            errors.append(msg)
            
            # Short sleep to prevent CPU spinning
            time.sleep(0.1)
            
        # End the progress line
        print(" Done!", flush=True)
        
        # Final statistics
        elapsed_time = time.time() - start_time
        
        # Only show error count if there are errors
        if error_count > 0:
            print(f"Completed with {error_count} errors.")
        
    return 0 if error_count == 0 else 1

def main():
    """Minimalist main entry point."""
    args = parse_arguments()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return 1
    
    # First scan the directory to get file list - minimal output
    print("Scanning files...", end="", flush=True)
    file_list = list(find_files_optimized(args.input_dir, args.recursive))
    print(f" Found {len(file_list)} files.")
    
    # Process the files in batches
    return process_batches(args, file_list)

if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows
    sys.exit(main())