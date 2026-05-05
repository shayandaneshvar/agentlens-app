#!/usr/bin/env python3
"""
Extract Ground Truth from Coding Agent Trajectories

This script processes coding agent trajectory files (chat-export-logs.json) and:
1. Generates PTAs from each trajectory
2. Merges PTAs to identify common patterns
3. Extracts ground truth for validation

This follows the same workflow as the original extract_ground_truth.py
but adapted for SWE/coding agent trajectories.

Usage:
    python extract_swe_ground_truth.py <trajectories_root> <instance_ids> [options]
    
Examples:
    # Generate PTA from single instance
    python extract_swe_ground_truth.py ./coding-agent-trajectories "run-21248319029-instance-chat_mode_simple-logs" --only-generate-pta
    
    # Generate and merge PTAs from multiple instances
    python extract_swe_ground_truth.py ./coding-agent-trajectories "run-21248319029-instance-chat_mode_simple-logs,run-21244897627-instance-chat_mode_simple-logs"
    
    # Process all instances in directory
    python extract_swe_ground_truth.py ./coding-agent-trajectories --all
    
    # Full pipeline with ground truth extraction
    python extract_swe_ground_truth.py ./coding-agent-trajectories --all --output-dir ./swe_outputs
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from swe_models import PTA, State, Transition
from swe_pta_generator import PTAGenerator, generate_pta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global cache directory for extracted ZIPs
_extract_cache_dir: Optional[Path] = None


def get_extract_cache_dir() -> Path:
    """Get or create the cache directory for extracted ZIP files."""
    global _extract_cache_dir
    if _extract_cache_dir is None:
        _extract_cache_dir = Path(tempfile.mkdtemp(prefix="swe_trajectory_cache_"))
        logger.info(f"Created extraction cache: {_extract_cache_dir}")
    return _extract_cache_dir


def cleanup_extract_cache():
    """Clean up the extraction cache directory."""
    global _extract_cache_dir
    if _extract_cache_dir and _extract_cache_dir.exists():
        try:
            shutil.rmtree(_extract_cache_dir)
            logger.info(f"Cleaned up extraction cache: {_extract_cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up cache: {e}")
        _extract_cache_dir = None


def extract_zip_file(zip_path: Path, cache_dir: Path = None) -> Optional[Path]:
    """
    Extract a ZIP file to a cache directory.
    
    Args:
        zip_path: Path to the ZIP file
        cache_dir: Directory to extract to (uses temp cache if None)
        
    Returns:
        Path to extracted directory, or None if extraction failed
    """
    if cache_dir is None:
        cache_dir = get_extract_cache_dir()
    
    # Create a subdirectory based on ZIP filename (without extension)
    extract_name = zip_path.stem
    extract_dir = cache_dir / extract_name
    
    # Skip if already extracted
    if extract_dir.exists():
        return extract_dir
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        logger.debug(f"Extracted: {zip_path.name} -> {extract_dir}")
        return extract_dir
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return None


def is_zip_file(path: Path) -> bool:
    """Check if a path is a ZIP file."""
    return path.is_file() and path.suffix.lower() == '.zip'


def get_pass_fail_status(name: str) -> Optional[str]:
    """
    Determine pass/fail status from a filename or directory name.
    
    Returns 'passed', 'failed', or None if undetermined.
    """
    name_lower = name.lower()
    if '-pass.' in name_lower or name_lower.endswith('-pass') or '-pass.zip' in name_lower:
        return 'passed'
    elif '-fail.' in name_lower or name_lower.endswith('-fail') or '-fail.zip' in name_lower:
        return 'failed'
    return None


def find_trajectory_file(instance_dir: Path) -> Optional[Path]:
    """
    Find the chat-export-logs.json file in an instance directory.
    
    Expected structure:
    instance_dir/
        output/
            vsc-output/
                chat-export-logs.json
    """
    # Try common paths
    candidates = [
        instance_dir / "output" / "vsc-output" / "chat-export-logs.json",
        instance_dir / "vsc-output" / "chat-export-logs.json",
        instance_dir / "chat-export-logs.json",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Search recursively for the file
    for path in instance_dir.rglob("chat-export-logs.json"):
        return path
    
    return None


def process_single_instance(
    trajectories_root: Path,
    instance_id: str,
    output_dir: Path,
    verbose: bool = False
) -> Optional[Tuple[Path, PTA]]:
    """
    Process a single instance and generate its PTA.
    
    Supports both:
    - Directory-based instances
    - ZIP file-based instances (will be extracted automatically)
    
    Args:
        trajectories_root: Root directory containing trajectory instances
        instance_id: Instance folder name or ZIP filename
        output_dir: Directory to save output files
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (path to generated PTA file, PTA object), or None if failed
    """
    instance_path = trajectories_root / instance_id
    
    # Handle ZIP files
    if is_zip_file(instance_path):
        logger.info(f"Extracting ZIP: {instance_id}")
        instance_dir = extract_zip_file(instance_path)
        if instance_dir is None:
            logger.error(f"Failed to extract: {instance_id}")
            return None
    elif instance_path.is_dir():
        instance_dir = instance_path
    else:
        logger.error(f"Instance not found: {instance_path}")
        return None
    
    # Find trajectory file
    trajectory_file = find_trajectory_file(instance_dir)
    if not trajectory_file:
        logger.error(f"No chat-export-logs.json found in: {instance_dir}")
        return None
    
    logger.info(f"Processing: {instance_id}")
    logger.info(f"  Trajectory file: {trajectory_file}")
    
    try:
        # Generate PTA
        generator = PTAGenerator()
        pta = generator.generate_pta(str(trajectory_file))
        
        # Add instance metadata
        pta.metadata["instance_id"] = instance_id
        pta.metadata["trajectory_path"] = str(trajectory_file)
        pta.metadata["pass_fail_status"] = get_pass_fail_status(instance_id)
        
        # Save PTA
        safe_instance_id = instance_id.replace("/", "_").replace("\\", "_").replace(".zip", "")
        output_file = output_dir / f"{safe_instance_id}_pta.json"
        pta.save(str(output_file))
        
        logger.info(f"  Generated PTA: {len(pta.states)} states, {len(pta.transitions)} transitions")
        logger.info(f"  Saved to: {output_file}")
        
        return (output_file, pta)
        
    except Exception as e:
        logger.error(f"  Error processing {instance_id}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def discover_instances(trajectories_root: Path, filter_status: str = None) -> List[str]:
    """
    Discover all instance directories or ZIP files in the trajectories root.
    
    Supports both:
    - Directory-based instances (existing format)
    - ZIP file-based instances (new format from evaluation platform)
    
    Args:
        trajectories_root: Root directory containing trajectory instances
        filter_status: Filter by pass/fail status. Options: 'passed', 'failed', None (all)
    
    Returns list of instance identifiers (directory names or ZIP filenames).
    """
    instances = []
    
    for item in trajectories_root.iterdir():
        # Skip hidden items
        if item.name.startswith('.'):
            continue
        
        # Handle ZIP files
        if is_zip_file(item):
            item_status = get_pass_fail_status(item.name)
            if filter_status is None or item_status == filter_status:
                instances.append(item.name)
            continue
        
        # Handle directories
        if item.is_dir():
            # Check if it contains a trajectory file
            if find_trajectory_file(item):
                item_status = get_pass_fail_status(item.name)
                if filter_status is None or item_status == filter_status:
                    instances.append(item.name)
    
    return sorted(instances)


def merge_ptas(
    generated_ptas: List[Tuple[str, Path, PTA]],
    output_dir: Path,
    use_llm: bool = True,
    verbose: bool = False
) -> Optional[Tuple[Path, PTA]]:
    """
    Merge multiple PTAs incrementally.
    
    Args:
        generated_ptas: List of (instance_id, pta_file, pta_object) tuples
        output_dir: Output directory
        use_llm: Whether to use LLM for equivalence checking
        verbose: Enable verbose output
        
    Returns:
        Tuple of (merged_pta_file, merged_pta) or None if failed
    """
    if len(generated_ptas) < 2:
        logger.warning("Need at least 2 PTAs to merge")
        if generated_ptas:
            return (generated_ptas[0][1], generated_ptas[0][2])
        return None
    
    try:
        from swe_pta_merger import SWEPTAMerger
    except ImportError as e:
        logger.error(f"Cannot import SWEPTAMerger: {e}")
        return None
    
    print(f"\n{'='*60}")
    print("🔧 MERGING PTAs")
    print(f"{'='*60}")
    
    merger = SWEPTAMerger(use_llm=use_llm, verbose=verbose)
    
    # Start with first PTA
    current_pta = generated_ptas[0][2]
    print(f"\n[1/{len(generated_ptas)}] Initial PTA: {generated_ptas[0][0]}")
    print(f"   States: {len(current_pta.states)}, Transitions: {len(current_pta.transitions)}")
    
    # Merge remaining PTAs incrementally
    for i, (instance_id, pta_file, pta) in enumerate(generated_ptas[1:], 2):
        print(f"\n[{i}/{len(generated_ptas)}] Merging: {instance_id}")
        
        try:
            ptas_to_merge = [current_pta, pta]
            current_pta = merger.merge_ptas(ptas_to_merge)
            
            print(f"   After merge: {len(current_pta.states)} states, {len(current_pta.transitions)} transitions")
            
        except Exception as e:
            logger.error(f"   Error merging {instance_id}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    # Save merged PTA
    merged_file = output_dir / "merged_pta.json"
    current_pta.save(str(merged_file))
    
    # Print merge stats
    stats = merger.get_stats()
    print(f"\n{'='*60}")
    print("MERGE STATISTICS")
    print(f"{'='*60}")
    for key, value in stats["merge_stats"].items():
        print(f"  {key}: {value}")
    print("\nEquivalence Stats:")
    for key, value in stats["equivalence_stats"].items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Merged PTA saved to: {merged_file}")
    
    return (merged_file, current_pta)


def generalize_pta(
    merged_pta: PTA,
    merged_file: Path,
    output_dir: Path,
    mode: str = "dominators",
    verbose: bool = False
) -> Optional[Tuple[Path, PTA]]:
    """
    Generalize merged PTA to extract ground truth.
    
    This uses the dominator analysis to identify essential states.
    
    Args:
        merged_pta: Merged PTA to generalize
        merged_file: Path to merged PTA file
        output_dir: Output directory
        mode: Generalization mode ('dominators' or 'shortest_path')
        verbose: Enable verbose output
        
    Returns:
        Tuple of (ground_truth_file, ground_truth_pta) or None if failed
    """
    print(f"\n{'='*60}")
    print("🌳 GENERALIZING TO GROUND TRUTH")
    print(f"{'='*60}")
    
    try:
        from pta_generalizer import PTAGeneralizer
    except ImportError as e:
        logger.warning(f"Cannot import PTAGeneralizer: {e}")
        print("   Skipping generalization (pta_generalizer not available)")
        return None
    
    print(f"   Mode: {mode}")
    print(f"   Input: {merged_file}")
    
    try:
        generalizer = PTAGeneralizer(verbose=verbose)
        ground_truth = generalizer.generalize(merged_pta, mode=mode)
        
        # Save ground truth
        gt_file = output_dir / "ground_truth.json"
        generalizer.save_pta(ground_truth, str(gt_file))
        
        # Print summary
        print(f"\n   Ground truth extracted:")
        print(f"   - States: {len(ground_truth.states)}")
        print(f"   - Transitions: {len(ground_truth.transitions)}")
        print(f"   - Terminal states: {len(ground_truth.get_terminal_states())}")
        print(f"\n✅ Ground truth saved to: {gt_file}")
        
        return (gt_file, ground_truth)
        
    except Exception as e:
        logger.error(f"   Error during generalization: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract ground truth from coding agent trajectories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('trajectories_root', 
                       help='Root directory containing trajectory instances')
    parser.add_argument('instance_ids', nargs='?', default=None,
                       help='Comma-separated list of instance folder names')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Process all instances in the root directory')
    parser.add_argument('--passed', '-p', action='store_true',
                       help='Process only passed trajectories (filename ends with -pass)')
    parser.add_argument('--failed', '-f', action='store_true',
                       help='Process only failed trajectories (filename ends with -fail)')
    parser.add_argument('--output-dir', '-d', default=None,
                       help='Output directory for generated files (default: <trajectories_root>/pta_outputs)')
    parser.add_argument('--only-generate-pta', action='store_true',
                       help='Only generate individual PTAs without merging')
    parser.add_argument('--no-generalize', action='store_true',
                       help='Skip generalization step (only merge PTAs)')
    parser.add_argument('--generalization-mode', choices=['dominators', 'shortest_path'],
                       default='dominators',
                       help='Generalization algorithm (default: dominators)')
    parser.add_argument('--no-llm', action='store_true',
                       help='Disable LLM-based state equivalence')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available instances and exit')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    trajectories_root = Path(args.trajectories_root)
    if not trajectories_root.exists():
        logger.error(f"Trajectories root not found: {trajectories_root}")
        sys.exit(1)
    
    # List mode
    if args.list:
        # Determine filter for list mode too
        if args.passed:
            filter_status = 'passed'
        elif args.failed:
            filter_status = 'failed'
        else:
            filter_status = None
        
        instances = discover_instances(trajectories_root, filter_status=filter_status)
        status_label = f" ({filter_status})" if filter_status else ""
        print(f"Found {len(instances)} instances{status_label} in {trajectories_root}:")
        for inst in instances:
            traj = find_trajectory_file(trajectories_root / inst)
            print(f"  - {inst}")
            if args.verbose and traj:
                print(f"      {traj}")
        sys.exit(0)
    
    # Determine instances to process
    if args.all or args.passed or args.failed:
        # Determine filter status
        if args.passed and args.failed:
            logger.error("Cannot use both --passed and --failed together. Use --all for all instances.")
            sys.exit(1)
        elif args.passed:
            filter_status = 'passed'
        elif args.failed:
            filter_status = 'failed'
        else:
            filter_status = None
        
        instance_ids = discover_instances(trajectories_root, filter_status=filter_status)
        if not instance_ids:
            status_msg = f" with status '{filter_status}'" if filter_status else ""
            logger.error(f"No instances found{status_msg} in trajectories root")
            sys.exit(1)
        
        status_label = f" ({filter_status})" if filter_status else ""
        logger.info(f"Discovered {len(instance_ids)} instances{status_label}")
    elif args.instance_ids:
        instance_ids = [i.strip() for i in args.instance_ids.split(',')]
    else:
        logger.error("Must specify instance_ids or use --all/--passed/--failed")
        parser.print_help()
        sys.exit(1)
    
    # Create output directory (default to inside trajectories_root)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = trajectories_root / "pta_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"SWE GROUND TRUTH EXTRACTION")
    print(f"{'='*60}")
    print(f"Processing {len(instance_ids)} instance(s)")
    print(f"Output directory: {output_dir}")
    print(f"LLM equivalence: {'disabled' if args.no_llm else 'enabled'}")
    
    # Phase 1: Generate PTAs
    print(f"\n{'='*60}")
    print("📥 PHASE 1: GENERATING PTAs")
    print(f"{'='*60}")
    
    generated_ptas: List[Tuple[str, Path, PTA]] = []
    failed_instances = []
    
    for i, instance_id in enumerate(instance_ids, 1):
        print(f"\n[{i}/{len(instance_ids)}] {instance_id}")
        
        result = process_single_instance(
            trajectories_root,
            instance_id,
            output_dir,
            args.verbose
        )
        
        if result:
            pta_file, pta = result
            generated_ptas.append((instance_id, pta_file, pta))
        else:
            failed_instances.append(instance_id)
    
    # Summary
    print(f"\n{'='*60}")
    print("PHASE 1 SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {len(generated_ptas)}/{len(instance_ids)}")
    
    if generated_ptas:
        print("\nGenerated PTAs:")
        for instance_id, pta_file, pta in generated_ptas:
            print(f"  - {pta_file.name} ({len(pta.states)} states)")
    
    if failed_instances:
        print(f"\nFailed ({len(failed_instances)}):")
        for inst in failed_instances:
            print(f"  - {inst}")
    
    # If only generating PTAs, we're done
    if args.only_generate_pta:
        print(f"\n✅ PTA generation complete (--only-generate-pta)")
        if failed_instances:
            sys.exit(1)
        sys.exit(0)
    
    # Phase 2: Merge PTAs
    if len(generated_ptas) >= 2:
        merge_result = merge_ptas(
            generated_ptas,
            output_dir,
            use_llm=not args.no_llm,
            verbose=args.verbose
        )
        
        if merge_result and not args.no_generalize:
            # Phase 3: Generalize to ground truth
            merged_file, merged_pta = merge_result
            generalize_pta(
                merged_pta,
                merged_file,
                output_dir,
                mode=args.generalization_mode,
                verbose=args.verbose
            )
    elif len(generated_ptas) == 1:
        print(f"\n⚠️  Only 1 PTA generated - skipping merge")
        print(f"   Use multiple instances for ground truth extraction")
    else:
        print(f"\n❌ No PTAs generated - cannot proceed")
        sys.exit(1)
    
    # Final summary
    print(f"\n{'='*60}")
    print("✅ EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    
    # List output files
    print("\nGenerated files:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file() and f.suffix == '.json':
            print(f"  - {f.name}")
    
    # Clean up extraction cache
    cleanup_extract_cache()
    
    # Exit with error if any failed
    if failed_instances:
        sys.exit(1)


if __name__ == "__main__":
    main()
