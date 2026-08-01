#!/usr/bin/env python3
"""
Convert PSpice .lib/.cir files to ngspice-compatible format.
Fixes .PARAM syntax by adding equals signs between parameter name and value.
"""

import re
import sys
from pathlib import Path


def convert_pspice_to_ngspice(input_path: Path, output_path: Path) -> None:
    """
    Convert PSpice parameter syntax to ngspice-compatible format.
    
    PSpice uses: .PARAM name value
    ngspice needs: .PARAM name = value
    
    Also handles case sensitivity issues in parameter references.
    """
    content = input_path.read_text(encoding='utf-8', errors='replace')
    
    # Normalize line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    lines = content.split('\n')
    
    converted_lines = []
    # Match .PARAM lines that don't already have '=' after the parameter name
    # Pattern 1: .PARAM name value (numeric value)
    # Pattern 2: .PARAM name {expression} (expression in braces)
    param_pattern = re.compile(r'^(\s*\.PARAM\s+)(\w+)(\s+)(.*)$')
    
    for line in lines:
        # Match .PARAM lines that don't already have '='
        match = param_pattern.match(line)
        if match:
            prefix = match.group(1)    # ".PARAM  "
            param_name = match.group(2)  # "ptrc1"
            spacing = match.group(3)    # "   "
            param_value = match.group(4).rstrip() # "5.5e-3" or "{pAA/0.8*2}"
            
            # Skip if already has equals
            if '=' in spacing or param_value.startswith('='):
                converted_lines.append(line)
                continue
                
            # Add equals sign
            new_line = f"{prefix}{param_name} = {param_value}"
            converted_lines.append(new_line)
            print(f"Fixed: {line.strip()} -> {new_line.strip()}")
        else:
            converted_lines.append(line)
    
    # Join with proper newlines
    content = '\n'.join(converted_lines)
    
    # Also fix case sensitivity: pWidth -> pwidth in expressions
    content = content.replace('pWidth', 'pwidth')
    
    output_path.write_text(content, encoding='utf-8')
    print(f"\nConverted file saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_spice.py <input_file.lib>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    # Generate output filename: ng_<original_name>
    output_file = input_file.parent / f"ng_{input_file.name}"
    
    convert_pspice_to_ngspice(input_file, output_file)
    
    print("\n" + "="*60)
    print("REMINDER: Ensure 'set ngbehavior=ps' is configured in your .spiceinit file")
    print("="*60)