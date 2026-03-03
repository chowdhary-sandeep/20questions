import json
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image
from typing import List

from world.scenarioGenerator import ScenarioConfig


def get_agent_colors(n_agents):
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    colors[:, 3] = 0.7  # consistent alpha for agents and goals
    return colors

def create_figure_and_ax(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)  # top-left origin
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

def draw_grid(ax, width, height):
    for x in range(width + 1):
        ax.axvline(x - 0.5, color='#CCCCCC', linewidth=0.5)
    for y in range(height + 1):
        ax.axhline(y - 0.5, color='#CCCCCC', linewidth=0.5)

def draw_obstacles(ax, obstacles):
    for (x, y) in obstacles or []:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                               facecolor='#000000', edgecolor='black', linewidth=1))

def draw_goals(ax, goal_positions, agent_colors, alpha=0.3):
    for i, g in enumerate(goal_positions or []):
        rgb = agent_colors[i][:3]
        ax.add_patch(Rectangle((g[0] - 0.4, g[1] - 0.4), 0.8, 0.8,
                               facecolor=(*rgb, alpha), edgecolor=rgb, linewidth=3))
        ax.text(g[0], g[1], 'G', ha='center', va='center',
                fontsize=12, fontweight='bold', color='black')

def draw_agents(ax, positions, agent_colors):
    for i, pos in enumerate(positions):
        color = agent_colors[i]
        ax.add_patch(Circle(pos, 0.35, facecolor=color, edgecolor='black', linewidth=2))
        ax.text(pos[0], pos[1], str(i), ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

def save_scenario_plot(scenario, filepath, show_coordinates=True, dpi=150):
    fig, ax = create_figure_and_ax(scenario.width, scenario.height)
    draw_grid(ax, scenario.width, scenario.height)
    if show_coordinates:
        for x in range(scenario.width):
            ax.text(x, -0.7, str(x), ha='center', va='top', fontsize=8, color='gray')
        for y in range(scenario.height):
            ax.text(-0.7, y, str(y), ha='right', va='center', fontsize=8, color='gray')
        ax.text(-0.7, -0.7, '(0,0)', ha='right', va='top', fontsize=8,
                color='red', fontweight='bold')
    draw_obstacles(ax, scenario.obstacles)
    n_agents = len(scenario.agent_positions)
    agent_colors = get_agent_colors(n_agents)
    draw_goals(ax, scenario.goal_positions, agent_colors)
    draw_agents(ax, scenario.agent_positions, agent_colors)
    ax.set_title(f"Scenario: {scenario.scenario_id} | Agents: {n_agents}", fontsize=12)
    plt.tight_layout()
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def save_paths_gif(agent_path_history, goal_positions, obstacles, width, height, filepath='agent_paths.gif', fps=2):
    n_agents = len(agent_path_history)
    if n_agents == 0:
        raise ValueError('No agent paths provided')

    agent_colors = get_agent_colors(n_agents)

    max_len = max(len(p) for p in agent_path_history)
    frames = []

    for t in range(max_len):
        fig, ax = create_figure_and_ax(width, height)
        draw_grid(ax, width, height)

        # Axis coordinate labels like scenario plot
        for x in range(width):
            ax.text(x, -0.7, str(x), ha='center', va='top', fontsize=8, color='gray')
        for y in range(height):
            ax.text(-0.7, y, str(y), ha='right', va='center', fontsize=8, color='gray')
        ax.text(-0.7, -0.7, '(0,0)', ha='right', va='top', fontsize=8,
                color='red', fontweight='bold')

        draw_obstacles(ax, obstacles)
        draw_goals(ax, goal_positions, agent_colors)

        for i, path in enumerate(agent_path_history):
            if not path:
                continue
            upto = min(t, len(path) - 1)
            pts = [tuple(p) for p in path[:upto + 1]]
            xs, ys = zip(*pts)
            if len(xs) > 1:
                ax.plot(xs, ys, '-', color=agent_colors[i], linewidth=2, alpha=0.7)
            ax.add_patch(Circle((xs[-1], ys[-1]), 0.35,
                                facecolor=agent_colors[i], edgecolor='black', linewidth=1.5))
            ax.text(xs[-1], ys[-1], str(i), ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).convert('RGBA')
        frames.append(img.copy())
        buf.close()
        plt.close(fig)

    if frames:
        duration = int(1000 / fps)
        frames[0].save(filepath, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        print(f'Saved agent paths GIF to {filepath}')
        return filepath
    return None


class ChatParser:
    def __init__(self, json_data=None, file_path=None, is_jsonl=False):
        if json_data is not None:
            self.data = json.loads(json_data) if isinstance(json_data, str) else json_data
        elif file_path is not None:
            self.data = self._load_jsonl(file_path) if is_jsonl else json.load(open(file_path, 'r', encoding='utf-8'))
        else:
            raise ValueError('Either json_data or file_path must be provided')

    def _load_jsonl(self, file_path):
        out = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    def _get_chat_by_id(self, chat_id):
        for chat in self.data:
            if chat.get('id') == chat_id:
                return chat
        return None

    def print_all_fields(self, chat_id):
        chat = self._get_chat_by_id(chat_id)
        if not chat:
            print(f'No chat found with ID: {chat_id}')
            return
        print(f'Chat ID: {chat_id}')
        print('Fields:')
        for k in chat.keys():
            print(' -', k)

    def get_info(self, chat_id):
        chat = self._get_chat_by_id(chat_id)
        return chat.get('info') if chat else None

    def get_completions(self, chat_id):
        chat = self._get_chat_by_id(chat_id)
        return chat.get('completion') if chat else None


def alternate_completions_as_list(completions: List[dict]):
    return [{'role': msg.get('role'), 'content': msg.get('content')} for msg in completions]


def parse_and_visualise(jsonl_path: str, chat_id: int, png_path: str, gif_path: str, completions_json_path: str):
    parser = ChatParser(file_path=jsonl_path, is_jsonl=True)
    parser.print_all_fields(chat_id)

    info = parser.get_info(chat_id)
    print('\nInfo:')
    print(json.dumps(info, indent=2, ensure_ascii=False))

    completions = parser.get_completions(chat_id) or []
    alt = alternate_completions_as_list(completions)
    with open(completions_json_path, 'w', encoding='utf-8') as f:
        json.dump(alt, f, indent=2, ensure_ascii=False)
    print(f'Wrote completions to {completions_json_path}')

    scenario_dict = info.get('original_scenario') if info and info.get('original_scenario') else (info or {})
    scenario_obj = None
    try:
        scenario_obj = ScenarioConfig.from_dict(scenario_dict)
    except Exception:
        class S: pass
        s = S()
        s.width = int(scenario_dict.get('width', 7))
        s.height = int(scenario_dict.get('height', 7))

        def _parse_list(lst):
            out = []
            for item in lst or []:
                if isinstance(item, str):
                    try:
                        out.append(json.loads(item.replace("'", '"')))
                    except Exception:
                        out.append(eval(item))
                else:
                    out.append(item)
            return out

        s.agent_positions = _parse_list(scenario_dict.get('agent_positions', []))
        s.goal_positions = _parse_list(scenario_dict.get('goal_positions', []))
        s.obstacles = _parse_list(scenario_dict.get('obstacles', []))
        s.scenario_id = scenario_dict.get('scenario_id', 'scenario')

        scenario_obj = s

    try:
        save_scenario_plot(scenario_obj, png_path)
        print(f'Saved scenario PNG to {png_path}')
    except Exception as e:
        print('Failed to save scenario PNG:', e)

    agent_path_history = info.get('agent_path_history') or info.get('agent_paths') or []
    if not agent_path_history:
        print('No agent path history found; skipping GIF creation')
        return

    cleaned = []
    for ap in agent_path_history:
        cleaned.append([(int(p[0]), int(p[1])) for p in ap])

    width = int(info.get('width', getattr(scenario_obj, 'width', 7)))
    height = int(info.get('height', getattr(scenario_obj, 'height', 7)))
    goals = info.get('goal_positions') or getattr(scenario_obj, 'goal_positions', [])
    obstacles = info.get('obstacles') or getattr(scenario_obj, 'obstacles', [])

    try:
        save_paths_gif(cleaned, goals, obstacles, width, height, filepath=gif_path, fps=2)
    except Exception as e:
        print('Failed to create GIF:', e)


if __name__ == '__main__':
    json_path = 'outputs/evals/MAPP--gpt-4.1/3f9314d8/results.jsonl'
    chat_id = 1
    parse_and_visualise(json_path, chat_id=chat_id,
                       png_path=f'outputs/evals/MAPP--gpt-4.1/3f9314d8/scenario_{chat_id}_overview.png',
                       gif_path=f'outputs/evals/MAPP--gpt-4.1/3f9314d8/agent_paths_{chat_id}.gif',
                       completions_json_path=f'outputs/evals/MAPP--gpt-4.1/3f9314d8/completions_chat_{chat_id}.json')
