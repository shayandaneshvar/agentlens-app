"""
PTA Visualization Generator

Creates interactive HTML visualizations using the SDK export API.
Generates:
1. Individual trajectory PTA visualizations
2. Merged PTA visualizations
3. Task index pages
4. Main dashboard

Usage:
    python create_visualizations.py <experiment_output_dir>
    
Example:
    python create_visualizations.py "C:\path\to\experiment_outputs"
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import html
from tqdm import tqdm

from swe_trace_sdk import trace as trace_api, export

# Configure logging - suppress verbose output
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Information about a task for visualization."""
    name: str
    folder: Path
    trajectories: List[Dict[str, Any]]
    merged_pta_path: Optional[Path]
    has_merged: bool


def parse_trajectory_name(filename: str) -> Dict[str, str]:
    """Parse trajectory filename to extract metadata."""
    name = filename.replace('_pta.json', '').replace('.json', '')
    
    status = 'unknown'
    if '-pass' in name:
        status = 'passed'
        name = name.replace('-pass', '')
    elif '-fail' in name:
        status = 'failed'
        name = name.replace('-fail', '')
    
    model = 'unknown'
    if '-logs-' in name:
        parts = name.split('-logs-')
        if len(parts) > 1:
            model = parts[1]
    
    return {
        'name': filename.replace('.json', ''),
        'status': status,
        'model': model
    }


def generate_task_index(task_info: TaskInfo, output_path: Path) -> None:
    """Generate an index page for a task showing all trajectories and merged PTA."""
    
    traj_cards = []
    for traj in task_info.trajectories:
        name = traj['name']
        status = traj.get('status', 'unknown')
        model = traj.get('model', 'unknown')
        visual_path = f"visuals/{name}_graph.html"
        
        status_color = '#3fb950' if status == 'passed' else '#f85149' if status == 'failed' else '#8b949e'
        
        traj_cards.append(f'''
        <a href="{visual_path}" class="traj-card">
            <div class="traj-status" style="background: {status_color};">{status.upper()}</div>
            <div class="traj-name">{html.escape(name)}</div>
            <div class="traj-model">{html.escape(model)}</div>
        </a>
        ''')
    
    merged_section = ''
    if task_info.has_merged:
        merged_section = '''
        <div class="section">
            <h2>🔀 Merged PTA</h2>
            <p>Combined PTA from all trajectories showing common patterns and variations.</p>
            <a href="visuals/merged_pta_graph.html" class="merged-link">View Merged PTA →</a>
        </div>
        '''
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(task_info.name)} - PTA Visualizations</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #58a6ff;
            margin-bottom: 8px;
            font-size: 1.8rem;
        }}
        .subtitle {{
            color: #8b949e;
            margin-bottom: 24px;
        }}
        .back-link {{
            display: inline-block;
            margin-bottom: 16px;
            color: #58a6ff;
            text-decoration: none;
        }}
        .back-link:hover {{
            text-decoration: underline;
        }}
        .section {{
            margin: 32px 0;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 24px;
        }}
        .section h2 {{
            color: #c9d1d9;
            border-bottom: 1px solid #30363d;
            padding-bottom: 8px;
            margin-bottom: 16px;
            font-size: 1.2rem;
        }}
        .traj-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 16px;
        }}
        .traj-card {{
            display: block;
            background: #21262d;
            border-radius: 8px;
            padding: 16px;
            text-decoration: none;
            color: inherit;
            transition: all 0.2s;
            border: 1px solid #30363d;
        }}
        .traj-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(88, 166, 255, 0.2);
            border-color: #58a6ff;
        }}
        .traj-status {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            color: white;
            font-size: 11px;
            font-weight: bold;
            margin-bottom: 8px;
        }}
        .traj-name {{
            font-weight: 600;
            color: #c9d1d9;
            margin-bottom: 4px;
            word-break: break-all;
            font-size: 0.85rem;
        }}
        .traj-model {{
            color: #8b949e;
            font-size: 12px;
        }}
        .merged-link {{
            display: inline-block;
            background: #238636;
            color: white;
            padding: 12px 24px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .merged-link:hover {{
            background: #2ea043;
        }}
        .stats-row {{
            display: flex;
            gap: 16px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: #21262d;
            padding: 12px 20px;
            border-radius: 8px;
            border: 1px solid #30363d;
        }}
        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #58a6ff;
        }}
        .stat-label {{
            font-size: 11px;
            color: #8b949e;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="../index.html" class="back-link">← Back to All Tasks</a>
        <h1>📊 {html.escape(task_info.name)}</h1>
        <p class="subtitle">PTA Visualizations for this task</p>
        
        <div class="stats-row">
            <div class="stat">
                <div class="stat-value">{len(task_info.trajectories)}</div>
                <div class="stat-label">Trajectories</div>
            </div>
            <div class="stat">
                <div class="stat-value">{sum(1 for t in task_info.trajectories if t.get('status') == 'passed')}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{sum(1 for t in task_info.trajectories if t.get('status') == 'failed')}</div>
                <div class="stat-label">Failed</div>
            </div>
        </div>
        
        {merged_section}
        
        <div class="section">
            <h2>📁 Individual Trajectories</h2>
            <div class="traj-grid">
                {''.join(traj_cards)}
            </div>
        </div>
    </div>
</body>
</html>
'''
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def generate_main_index(tasks: List[TaskInfo], output_path: Path) -> None:
    """Generate the main index page listing all tasks."""
    
    task_cards = []
    for task in sorted(tasks, key=lambda t: t.name):
        passed = sum(1 for t in task.trajectories if t.get('status') == 'passed')
        failed = sum(1 for t in task.trajectories if t.get('status') == 'failed')
        total = len(task.trajectories)
        
        merged_badge = '<span class="badge merged">MERGED</span>' if task.has_merged else ''
        
        task_cards.append(f'''
        <a href="{task.name}/index.html" class="task-card">
            <div class="task-header">
                <div class="task-name">{html.escape(task.name)}</div>
                {merged_badge}
            </div>
            <div class="task-stats">
                <span class="stat-pill passed">{passed} passed</span>
                <span class="stat-pill failed">{failed} failed</span>
                <span class="stat-pill total">{total} total</span>
            </div>
        </a>
        ''')
    
    total_tasks = len(tasks)
    total_trajs = sum(len(t.trajectories) for t in tasks)
    total_passed = sum(sum(1 for tr in t.trajectories if tr.get('status') == 'passed') for t in tasks)
    total_merged = sum(1 for t in tasks if t.has_merged)
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PTA Experiment Visualizations</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            min-height: 100vh;
        }}
        .header {{
            background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
            border-bottom: 1px solid #30363d;
            padding: 48px 24px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 2rem;
            color: #58a6ff;
        }}
        .header p {{
            margin: 0;
            color: #8b949e;
        }}
        .summary-stats {{
            display: flex;
            justify-content: center;
            gap: 32px;
            margin-top: 32px;
            flex-wrap: wrap;
        }}
        .summary-stat {{
            text-align: center;
            background: #0d1117;
            padding: 16px 24px;
            border-radius: 8px;
            border: 1px solid #30363d;
        }}
        .summary-stat .value {{
            font-size: 32px;
            font-weight: bold;
            color: #58a6ff;
        }}
        .summary-stat .label {{
            font-size: 12px;
            color: #8b949e;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px 24px;
        }}
        .search-box {{
            width: 100%;
            max-width: 400px;
            margin: 0 auto 32px;
            display: block;
        }}
        .search-box input {{
            width: 100%;
            padding: 14px 20px;
            font-size: 14px;
            border: 1px solid #30363d;
            border-radius: 8px;
            background: #161b22;
            color: #c9d1d9;
            outline: none;
        }}
        .search-box input:focus {{
            border-color: #58a6ff;
        }}
        .search-box input::placeholder {{
            color: #8b949e;
        }}
        .task-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 16px;
        }}
        .task-card {{
            display: block;
            background: #161b22;
            border-radius: 8px;
            padding: 20px;
            text-decoration: none;
            color: #c9d1d9;
            transition: all 0.2s;
            border: 1px solid #30363d;
        }}
        .task-card:hover {{
            transform: translateY(-2px);
            border-color: #58a6ff;
            box-shadow: 0 4px 16px rgba(88, 166, 255, 0.15);
        }}
        .task-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }}
        .task-name {{
            font-size: 1rem;
            font-weight: 600;
            word-break: break-word;
            color: #c9d1d9;
        }}
        .badge {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
        }}
        .badge.merged {{
            background: #238636;
            color: white;
        }}
        .task-stats {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .stat-pill {{
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 500;
        }}
        .stat-pill.passed {{
            background: rgba(63, 185, 80, 0.15);
            color: #3fb950;
        }}
        .stat-pill.failed {{
            background: rgba(248, 81, 73, 0.15);
            color: #f85149;
        }}
        .stat-pill.total {{
            background: rgba(139, 148, 158, 0.15);
            color: #8b949e;
        }}
        .no-results {{
            text-align: center;
            color: #8b949e;
            padding: 48px;
            display: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 PTA Experiment Results</h1>
        <p>Interactive visualizations of Prefix Tree Acceptors from coding agent trajectories</p>
        
        <div class="summary-stats">
            <div class="summary-stat">
                <div class="value">{total_tasks}</div>
                <div class="label">Tasks</div>
            </div>
            <div class="summary-stat">
                <div class="value">{total_trajs}</div>
                <div class="label">Trajectories</div>
            </div>
            <div class="summary-stat">
                <div class="value">{total_passed}</div>
                <div class="label">Passed</div>
            </div>
            <div class="summary-stat">
                <div class="value">{total_merged}</div>
                <div class="label">Merged PTAs</div>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="search-box">
            <input type="text" id="search" placeholder="🔍 Search tasks..." oninput="filterTasks()">
        </div>
        
        <div class="task-grid" id="task-grid">
            {''.join(task_cards)}
        </div>
        
        <div class="no-results" id="no-results">
            No tasks found matching your search.
        </div>
    </div>
    
    <script>
        function filterTasks() {{
            const query = document.getElementById('search').value.toLowerCase();
            const cards = document.querySelectorAll('.task-card');
            let visibleCount = 0;
            
            cards.forEach(card => {{
                const name = card.querySelector('.task-name').textContent.toLowerCase();
                if (name.includes(query)) {{
                    card.style.display = 'block';
                    visibleCount++;
                }} else {{
                    card.style.display = 'none';
                }}
            }});
            
            document.getElementById('no-results').style.display = visibleCount === 0 ? 'block' : 'none';
        }}
    </script>
</body>
</html>
'''
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def process_experiment_output(experiment_dir: Path) -> List[TaskInfo]:
    """Process experiment output directory and gather task information."""
    tasks = []
    
    for task_folder in experiment_dir.iterdir():
        if not task_folder.is_dir():
            continue
        if task_folder.name.startswith('_') or task_folder.name.startswith('.'):
            continue
        
        pta_folder = task_folder / 'pta_outputs'
        if not pta_folder.exists():
            continue
        
        trajectories = []
        merged_pta_path = None
        
        for pta_file in pta_folder.glob('*.json'):
            if 'merged' in pta_file.name.lower():
                merged_pta_path = pta_file
            else:
                traj_info = parse_trajectory_name(pta_file.name)
                traj_info['path'] = pta_file
                trajectories.append(traj_info)
        
        if trajectories or merged_pta_path:
            tasks.append(TaskInfo(
                name=task_folder.name,
                folder=task_folder,
                trajectories=trajectories,
                merged_pta_path=merged_pta_path,
                has_merged=merged_pta_path is not None
            ))
    
    return tasks


def create_visualizations(experiment_dir: Path) -> None:
    """Create all visualizations for an experiment."""
    print(f"\n{'='*60}")
    print("PTA VISUALIZATION GENERATOR")
    print(f"{'='*60}")
    print(f"Source: {experiment_dir}")
    
    # Gather task information
    tasks = process_experiment_output(experiment_dir)
    
    if not tasks:
        print("❌ No tasks found in experiment directory")
        return
    
    # Count total visualizations to generate
    total_visuals = sum(len(t.trajectories) + (1 if t.has_merged else 0) for t in tasks)
    total_visuals += len(tasks)  # Task index pages
    total_visuals += 1  # Main index
    
    print(f"Tasks: {len(tasks)} | Trajectories: {sum(len(t.trajectories) for t in tasks)} | Total files: {total_visuals}")
    print(f"{'='*60}\n")
    
    # Suppress visualizer logging
    logging.getLogger('swe_trace_sdk').setLevel(logging.WARNING)
    
    # Progress bar for all tasks
    with tqdm(total=total_visuals, desc="Generating", unit="file", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        
        for task in tasks:
            visuals_dir = task.folder / 'visuals'
            visuals_dir.mkdir(exist_ok=True)
            
            # Generate visualizations for each trajectory using SDK export
            for traj in task.trajectories:
                pta_path = traj['path']
                output_path = visuals_dir / f"{traj['name']}_graph.html"
                
                try:
                    tr = trace_api.load(str(pta_path), format="trace")
                    export.trace(tr, str(output_path), format="html")
                except Exception as e:
                    pass  # Silently skip errors
                
                pbar.update(1)
            
            # Generate merged PTA visualization
            if task.merged_pta_path:
                output_path = visuals_dir / 'merged_pta_graph.html'
                try:
                    tr = trace_api.load(str(task.merged_pta_path), format="trace")
                    export.trace(tr, str(output_path), format="html")
                except Exception as e:
                    pass  # Silently skip errors
                
                pbar.update(1)
            
            # Generate task index
            generate_task_index(task, task.folder / 'index.html')
            pbar.update(1)
        
        # Generate main index
        generate_main_index(tasks, experiment_dir / 'index.html')
        pbar.update(1)
    
    print(f"\n{'='*60}")
    print("✅ VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"📂 Open: {experiment_dir / 'index.html'}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate PTA visualizations from experiment output'
    )
    parser.add_argument('experiment_dir', type=str,
                       help='Path to experiment output directory')
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"❌ Directory not found: {experiment_dir}")
        return 1
    
    create_visualizations(experiment_dir)
    return 0


if __name__ == "__main__":
    exit(main())
