import numpy as np
import math
import random
import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit


locations = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
actions = ["up", "down", "left", "right"]
target = "G"
actions_valid = {"A": ["right"], "B": ["left", "down", "right"], "C": ["left", "down"], "D": ["down"], "E": ["down"], "F": ["down", "up"], "G": ["up", "right"], "H": ["left", "up", "down"], "I": ["up", "right"], "J": ["up", "left", "right"], "K": ["left", "right"], "L": ["up", "left"]}
Qtable = np.zeros((len(locations), len(actions)))


def reward(next_state, valid : bool):
    if not valid:
        return -5
    elif next_state == target:
        return 100
    else:
        return -1


def position(state : str):
    return (locations.index(state) // 4, locations.index(state) % 4)


def move(state, action):
    if action not in actions_valid[state]:
        return state, reward(state, False)
    else:
        if action == "up":
            next_state = locations[locations.index(state) - 4]
        elif action == "down":
            next_state = locations[locations.index(state) + 4]
        elif action == "left":
            next_state = locations[locations.index(state) - 1]
        elif action == "right":
            next_state = locations[locations.index(state) + 1]
        return next_state, reward(next_state, True)


def check_target(state):
    return state == target

def explore(state):
    return random.choice(actions)


def exploit(state):
    return actions[np.argmax(Qtable[locations.index(state)])]


def choose_action(state, epsilon):
    randomise = random.random()
    if randomise < epsilon:
        return explore(state)
    else:
        return exploit(state)


def update_q(state, action, reward, new_state, alpha, gamma):
    if new_state == target:
        cible = reward
    else:
        cible = reward + gamma * (np.max(Qtable[locations.index(new_state)]))
    error = cible - Qtable[locations.index(state)][actions.index(action)]
    Qnouveau = Qtable[locations.index(state)][actions.index(action)] + alpha * error
    Qtable[locations.index(state)][actions.index(action)] = Qnouveau


def episode(start_state, alpha, gamma, epsilon, max_steps):
    current_state = start_state
    score = 0
    nb_steps = 0
    for step in range(max_steps):
        action = choose_action(current_state, epsilon)
        new_state, r = move(current_state, action)
        update_q(current_state, action, r, new_state, alpha, gamma)
        current_state = new_state
        nb_steps += 1
        if check_target(current_state):
            score += r
            break

        score += r

    return current_state == target, score, nb_steps


def train(nb_episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2,
          max_steps=200, on_episode=None):
    historique = []
    for episode_num in range(nb_episodes):
        start_state = random.choice([loc for loc in locations if loc != target])
        success, score, nb_steps = episode(start_state, alpha, gamma, epsilon, max_steps=max_steps)
        historique.append((success, score, nb_steps, start_state))
        if on_episode is not None:
            on_episode(episode_num + 1, Qtable)
    return historique


def evaluate(start_state, max_steps):
    current_state = start_state
    score = 0
    nb_steps = 0
    epsilon = 0
    chemin = []
    chemin.append(current_state)
    for step in range(max_steps):
        action = choose_action(current_state, epsilon)
        new_state, r = move(current_state, action)
        current_state = new_state
        nb_steps += 1
        if check_target(current_state):
            chemin.append(current_state)
            score += r
            break
        score += r
        chemin.append(current_state)
    
    return current_state == target, score, chemin



DEFAULT_CONFIG = {
    "episodes": 500,
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 0.2,
    "max_steps": 200,
    "seed": 42,
}


def validate_config(values):
    if not isinstance(values, dict):
        raise ValueError("Les paramètres doivent former un objet JSON.")
    unknown = set(values) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError("Paramètres inconnus : " + ", ".join(sorted(unknown)))
    config = {**DEFAULT_CONFIG, **values}
    for name, lower, upper in [
        ("episodes", 1, 5000),
        ("max_steps", 1, 2000),
        ("seed", 0, 2**32 - 1),
    ]:
        value = config[name]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} doit être un entier entre {lower} et {upper}.")
    for name in ("alpha", "gamma", "epsilon"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} doit être un nombre fini.")
        if not 0 <= value <= 1 or (name == "alpha" and value == 0):
            interval = "]0, 1]" if name == "alpha" else "[0, 1]"
            raise ValueError(f"{name} doit être dans {interval}.")
        config[name] = float(value)
    if config["episodes"] * config["max_steps"] > 2_000_000:
        raise ValueError("Réduis le nombre d’épisodes ou de pas : au maximum 2 000 000 transitions par entraînement.")
    return config


def run_visual_training(config=None):
    config = validate_config({} if config is None else config)
    random.seed(config["seed"])
    Qtable.fill(0)

    def snapshot():
        return {
            "q": Qtable.tolist(),
            "policy": np.argmax(Qtable, axis=1).tolist(),
        }

    snapshots = [snapshot()]

    def save_episode(episode_num, table):
        snapshots.append(snapshot())

    historique = train(
        nb_episodes=config["episodes"],
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon"],
        max_steps=config["max_steps"],
        on_episode=save_episode,
    )
    transitions = []
    for state in locations:
        row = []
        for action in actions:
            next_state, r = move(state, action)
            row.append({"next": next_state, "reward": r, "valid": action in actions_valid[state]})
        transitions.append(row)
    return {
        "config": config,
        "locations": locations,
        "actions": actions,
        "target": target,
        "transitions": transitions,
        "snapshots": snapshots,
        "history": [
            {"success": bool(ok), "score": score, "steps": steps, "start": start}
            for ok, score, steps, start in historique
        ],
    }


class VisualizationServer(HTTPServer):
    def __init__(self, address, initial_data):
        self.training_data = initial_data
        super().__init__(address, VisualizationHandler)


class VisualizationHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_content(self, status, content, content_type):
        payload = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, status, value):
        self.send_content(status, json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")), "application/json")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.send_content(200, HTML_INTERFACE, "text/html")
        elif path == "/api/data":
            self.send_json(200, self.server.training_data)
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_json(404, {"error": "Page inconnue."})

    def do_POST(self):
        if urlsplit(self.path).path != "/api/train":
            self.send_json(404, {"error": "Action inconnue."})
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Le corps de la requête doit être du JSON.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Taille de requête invalide.")
            config = validate_config(json.loads(self.rfile.read(length)))
        except (ValueError, UnicodeDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        self.server.training_data = run_visual_training(config)
        self.send_json(200, self.server.training_data)


def main():
    parser = argparse.ArgumentParser(description="TP1 : Q-learning avec interface interactive locale.")
    parser.add_argument("--port", type=int, default=0, help="Port local ; 0 choisit un port disponible.")
    parser.add_argument("--no-browser", action="store_true", help="Afficher l'URL sans ouvrir le navigateur.")
    args = parser.parse_args()
    with VisualizationServer(("127.0.0.1", args.port), run_visual_training()) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"URL: {url}", flush=True)
        print("Visualisation TP1 lancée. Ctrl+C pour arrêter.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nVisualisation arrêtée.")


HTML_INTERFACE = r"""
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>TP1 · Voir le Q-learning apprendre</title>
  <style>
    :root{color-scheme:light dark;--bg:#f5f6fa;--surface:#fff;--soft:#f0f3f8;--ink:#192437;--muted:#66758b;--line:#dce3ed;--blue:#2260d5;--blue-soft:#eaf1ff;--purple:#7856d8;--green:#13856a;--danger:#bf4149;--shadow:0 6px 25px #26375106}
    @media(prefers-color-scheme:dark){:root{--bg:#10151e;--surface:#18202d;--soft:#202b3b;--ink:#e5ecf8;--muted:#9daec5;--line:#354257;--blue:#7eabff;--blue-soft:#223858;--purple:#b19aff;--green:#54c9a6;--danger:#ff9196;--shadow:none}}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,select{font:inherit}button,select,input{color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:9px}button{padding:9px 14px;cursor:pointer;font-weight:600}button:hover:not(:disabled){border-color:var(--blue);background:var(--blue-soft)}button:disabled{opacity:.5;cursor:wait}button:focus-visible,input:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid var(--blue);outline-offset:3px}.primary{background:var(--blue);color:var(--surface);border-color:var(--blue)}.primary:hover:not(:disabled){color:var(--ink)}input,select{min-width:0;padding:8px 10px}input[type=range]{padding:0;accent-color:var(--blue);width:100%}input[type=number]{width:100%}h1,h2,p{margin:0}h1{font-size:clamp(25px,4vw,38px);line-height:1.2;letter-spacing:-1px}h2{font-size:18px;letter-spacing:-.3px}.eyebrow{font-size:11px;letter-spacing:1.8px;font-weight:750;text-transform:uppercase;color:var(--blue);margin-bottom:9px}.muted{color:var(--muted)}.small{font-size:12px}.app{max-width:1380px;margin:auto;padding:34px 28px 42px}.intro{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:24px}.intro p{margin-top:10px;color:var(--muted)}.tag{display:inline-flex;align-items:center;gap:7px;white-space:nowrap;border:1px solid var(--line);border-radius:30px;padding:6px 11px;font-size:12px}.tag:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--green)}.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);padding:20px}.controls{margin-bottom:18px}.card-heading{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:16px}.hint{color:var(--muted);font-size:12px}.parameter-grid{display:grid;grid-template-columns:repeat(6,minmax(75px,1fr)) auto;gap:12px;align-items:end}label{display:block;font-size:12px;font-weight:600}label input,label select{display:block;margin-top:6px}label .hint{font-size:11px;font-weight:400;display:block;min-height:17px;margin-top:4px}.parameter-grid button{margin-bottom:21px;min-height:40px}.parameters-note{margin-top:10px;font-size:12px;color:var(--muted)}.timeline{display:grid;grid-template-columns:auto minmax(100px,1fr) 95px;align-items:center;gap:18px;margin-top:20px;padding-top:18px;border-top:1px solid var(--line)}.timeline label{font-size:11px;color:var(--muted)}.timeline-meta{font-size:12px;color:var(--muted);margin-top:8px;min-height:18px}.dashboard{display:grid;grid-template-columns:minmax(0,1.12fr) minmax(0,1fr);gap:18px}.map-controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.map-controls label{display:flex;align-items:center;gap:8px}.map-controls select{margin:0}.map-controls button{padding:7px 12px}.map{display:block;width:100%;height:auto;max-height:405px}.map text{font-family:inherit}.corridor{stroke:var(--line);stroke-width:15;stroke-linecap:round}.corridor-core{stroke:var(--surface);stroke-width:7;stroke-linecap:round}.route-line{fill:none;stroke:var(--purple);stroke-width:4;stroke-linecap:round;stroke-linejoin:round;opacity:.72}.room{fill:var(--surface);stroke:var(--line);stroke-width:2}.room.start{stroke:var(--purple);stroke-width:3}.room.target{fill:var(--blue-soft);stroke:var(--green);stroke-width:2}.room-label{fill:var(--ink);font-size:20px;font-weight:700}.room-value{fill:var(--muted);font-size:11px}.policy-arrow{stroke:var(--blue);stroke-width:2.5;fill:none;stroke-linecap:round;stroke-linejoin:round}.map-key{display:flex;gap:16px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:-4px}.map-key i{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:5px;background:var(--blue);border-radius:3px}.map-key .purple{background:var(--purple)}.evaluation{margin-top:15px;background:var(--soft);border-radius:12px;padding:13px 15px}.evaluation-header{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:7px}.status{color:var(--green);font-size:12px;font-weight:700}.status.failed{color:var(--danger)}#route-text{font-size:15px;font-weight:600;overflow-wrap:anywhere}#evaluation-summary{margin-top:5px;font-size:12px;color:var(--muted)}.table-wrap{overflow:auto}table{width:100%;border-spacing:5px 4px;font-variant-numeric:tabular-nums;font-size:12px}th{color:var(--muted);font-size:11px;font-weight:600;white-space:nowrap}th[scope=row]{color:var(--ink);text-align:left;width:52px}td{padding:0;text-align:center}td button{font-size:12px;width:100%;min-width:64px;padding:6px 4px;border-radius:6px;border-color:transparent;font-variant-numeric:tabular-nums;font-weight:450}td button.greedy{font-weight:750}td button.selected{outline:2px solid var(--purple);outline-offset:0}.terminal-label{color:var(--green);font-size:10px;display:block}#q-description{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-top:14px;padding-top:14px;border-top:1px solid var(--line)}#q-value{font-size:23px;font-weight:700;letter-spacing:-.5px}.chart-card{margin-top:18px}.charts{display:grid;grid-template-columns:1fr 1fr;gap:24px}.chart-block{min-width:0}.chart-head{display:flex;align-items:baseline;gap:10px;justify-content:space-between}.chart-title{font-size:14px;font-weight:650}.chart{display:block;width:100%;height:auto}.chart-key{font-size:11px;color:var(--muted);display:flex;gap:14px;flex-wrap:wrap}.chart-key span:before{content:"";display:inline-block;width:16px;height:3px;vertical-align:middle;margin-right:5px;background:var(--blue)}.chart-key span.raw:before{opacity:.25}.chart-key span.reference:before{height:0;border-top:2px dashed var(--green);background:none}.chart-note{margin-top:6px;font-size:11px;color:var(--muted)}.error{padding:12px 16px;margin-bottom:18px;color:var(--danger);border:1px solid currentColor;border-radius:10px;background:var(--surface)}[hidden]{display:none!important}.footer{margin-top:17px;font-size:11px;color:var(--muted);text-align:center}.live-message{min-height:18px;font-size:12px;color:var(--muted);margin-top:6px}.stats{display:flex;gap:13px;font-size:12px}.stats strong{color:var(--ink)}
    @media(max-width:1000px){.parameter-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.parameter-grid button{margin-bottom:0}.dashboard{grid-template-columns:1fr 1fr}.card{padding:16px}.app{padding:24px 18px}td button{min-width:54px;font-size:11px}.stats{flex-direction:column;gap:0}}
    @media(max-width:760px){.dashboard,.charts{grid-template-columns:1fr}.intro{display:block}.intro .tag{margin-top:14px}.parameter-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.timeline{gap:10px;grid-template-columns:auto minmax(50px,1fr) 76px}.timeline button{padding:9px}.card-heading{align-items:flex-start}.app{padding:20px 12px}.charts{gap:26px}.chart-card{padding:14px}.map{max-height:360px}.stats{flex-direction:row}td button{min-width:45px}}
    @media(max-width:480px){.card-heading{flex-wrap:wrap}.map-controls{width:100%;justify-content:space-between}.chart-head{flex-wrap:wrap;gap:2px}.parameter-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
    .map{max-height:none}
    .chart-head{flex-wrap:wrap}
    @media(max-width:480px){.controls>.card-heading{flex-direction:column;gap:8px}.stats{flex-wrap:wrap;gap:12px}.chart-head{gap:4px}}
    @media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
  </style>
</head>
<body>
<main class="app">
  <header class="intro"><div><div class="eyebrow">IA avancée · TP1</div><h1>Q-learning</h1><p>Algorithme, table et décisions du robot, épisode après épisode.</p></div><span class="tag">Exécution locale</span></header>
  <div id="error-message" class="error" role="alert" hidden></div>
  <section class="card controls" aria-labelledby="training-title">
    <div class="card-heading"><h2 id="training-title">Entraînement</h2><div id="training-stats" class="stats"></div></div>
    <form id="training-form">
      <div class="parameter-grid">
        <label>Épisodes<input name="episodes" type="number" min="1" max="5000" step="1" value="500" required><span class="hint">Nombre d'essais</span></label>
        <label>α · apprentissage<input name="alpha" type="number" min="0.0000000001" max="1" step="any" value="0.1" required><span class="hint">Poids de la correction</span></label>
        <label>γ · actualisation<input name="gamma" type="number" min="0" max="1" step="any" value="0.9" required><span class="hint">Poids du futur</span></label>
        <label>ε · exploration<input name="epsilon" type="number" min="0" max="1" step="any" value="0.2" required><span class="hint">Probabilité d'explorer</span></label>
        <label>Limite de pas<input name="max_steps" type="number" min="1" max="2000" step="1" value="200" required><span class="hint">Par épisode / évaluation</span></label>
        <label>Graine aléatoire<input name="seed" type="number" min="0" max="4294967295" step="1" value="42" required><span class="hint">Pour reproduire un essai</span></label>
        <button id="train-button" class="primary" type="submit">Entraîner</button>
      </div>
    </form>
    <p class="parameters-note">Chaque nouvel entraînement repart d'une Q-table à zéro. Les départs sont tirés au hasard, hors G.</p>
    <div class="timeline"><button id="play-episodes" type="button" disabled>▶ Lire</button><div><input id="episode-slider" type="range" min="0" max="500" value="500" step="1" aria-label="Épisode affiché" disabled><div class="hint">0 · avant apprentissage <span id="timeline-end" style="float:right">500</span></div></div><label>Épisode<input id="episode-number" type="number" min="0" max="500" value="500" step="1" disabled></label></div>
    <div id="episode-detail" class="timeline-meta" aria-live="polite">Chargement de l'entraînement…</div>
  </section>
  <div class="dashboard">
    <section class="card" aria-labelledby="map-title">
      <div class="card-heading"><div><h2 id="map-title">Le robot et sa politique</h2><p class="hint">Évaluation : ε = 0 · Q-table figée</p></div><div class="map-controls"><label>Départ<select id="start-state" disabled></select></label><button id="play-route" type="button" disabled>▶ Trajet</button></div></div>
      <svg id="map" class="map" viewBox="0 0 620 400" role="img" aria-label="Carte de l'entrepôt, actions gloutonnes et trajet évalué"></svg>
      <div class="map-key"><span><i></i>Action gloutonne</span><span><i class="purple"></i>Trajet évalué</span><span>Nombre sous l'état : max Q</span></div>
      <div class="evaluation"><div class="evaluation-header"><span class="small muted">Trajet à cet épisode</span><span id="route-status" class="status"></span></div><div id="route-text">—</div><div id="evaluation-summary">—</div><div id="route-progress" class="live-message" aria-live="polite"></div></div>
    </section>
    <section class="card" aria-labelledby="q-title">
      <div class="card-heading"><div><h2 id="q-title">Q-table</h2><p class="hint">Clique une case pour suivre son estimation.</p></div><span class="hint">★ action choisie</span></div>
      <div class="table-wrap"><table id="q-table"><thead><tr><th scope="col">État</th><th scope="col">↑ up</th><th scope="col">↓ down</th><th scope="col">← left</th><th scope="col">→ right</th></tr></thead><tbody></tbody></table></div>
      <div id="q-description"><span id="q-label">Q(A, right)</span><strong id="q-value">—</strong></div>
      <p id="policy-detail" class="hint" style="margin-top:6px"></p>
      <p class="hint" style="margin-top:9px">En cas d'égalité exacte, <code>np.argmax</code> retient la première action dans l'ordre up, down, left, right.</p>
    </section>
  </div>
  <section class="card chart-card" aria-label="Courbes d'apprentissage"><div class="charts">
    <div class="chart-block"><div class="chart-head"><span class="chart-title">Scores pendant l'entraînement</span><span class="hint">Somme non actualisée</span></div><svg id="score-chart" class="chart" viewBox="0 0 680 240" role="img" aria-label="Scores par épisode et moyenne mobile sur vingt épisodes"></svg><div class="chart-key"><span class="raw">Score de l'épisode</span><span>Moyenne mobile · 20</span></div><p class="chart-note">Départs variables et exploration active : les scores ne mesurent pas seuls la qualité de la politique.</p></div>
    <div class="chart-block"><div class="chart-head"><span id="q-chart-title" class="chart-title">Évolution de Q(A, right)</span><span class="hint">Retour actualisé estimé</span></div><svg id="q-chart" class="chart" viewBox="0 0 680 240" role="img" aria-label="Évolution de la case Q sélectionnée"></svg><div class="chart-key"><span>Estimation apprise</span><span id="reference-key" class="reference">Référence du trajet A → B → C → G</span></div><p id="q-chart-note" class="chart-note"></p></div>
  </div></section>
  <p class="footer">Le déplacement final vers G rapporte +100 ; les autres déplacements valides −1 ; une action bloquée −5. G termine l'épisode.</p>
</main>
<script>
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const glyphs = ['↑','↓','←','→'];
  const form = $('training-form');
  let data = null, episode = 0, start = 'A', selection = [0,3];
  let episodeTimer = null, routeTimer = null, routeStep = 0, evaluation = null;
  let busy = false, mapWidth = 620, mapCompact = false;
  const chartMetrics = new Map();
  const fmt = (number, digits = 2) => Number(number).toLocaleString('fr-FR', {maximumFractionDigits:digits,minimumFractionDigits:digits});
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const point = index => {
    const padding=mapCompact?34:54, firstY=mapCompact?50:60, rowGap=mapCompact?112:128;
    return [padding+(index%4)*(mapWidth-2*padding)/3,firstY+Math.floor(index/4)*rowGap];
  };
  const robotPoint = index => {const [x,y]=point(index);return[x+(mapCompact?15:20),y-(mapCompact?15:20)];};
  const snap = () => data.snapshots[episode];
  function showError(message){$('error-message').textContent=message;$('error-message').hidden=!message;}
  function stopRoute(){clearInterval(routeTimer);routeTimer=null;routeStep=0;$('play-route').textContent='▶ Trajet';$('route-progress').textContent='';}
  function stopEpisodes(){clearInterval(episodeTimer);episodeTimer=null;$('play-episodes').textContent='▶ Lire';}
  function stopAll(){stopRoute();stopEpisodes();}
  function setBusy(value){busy=value;form.querySelectorAll('input,button').forEach(el=>el.disabled=value);['episode-slider','episode-number','play-episodes','start-state','play-route'].forEach(id=>$(id).disabled=value||!data);$('train-button').textContent=value?'Entraînement…':'Entraîner';}
  async function request(url,options){const response=await fetch(url,options);let result;try{result=await response.json();}catch(error){throw new Error('Le serveur a renvoyé une réponse illisible.');}if(!response.ok)throw new Error(result.error||`Erreur HTTP ${response.status}`);return result;}
  function installData(next){
    if(!next||!next.snapshots||!next.snapshots.length||!next.history)throw new Error("Les données d'entraînement sont incomplètes.");
    stopAll();data=next;episode=data.snapshots.length-1;
    for(const [name,value] of Object.entries(data.config)){if(form.elements[name])form.elements[name].value=value;}
    $('start-state').innerHTML=data.locations.filter(s=>s!==data.target).map(s=>`<option value="${esc(s)}">${esc(s)}</option>`).join('');
    if(start===data.target||!data.locations.includes(start))start=data.locations.find(s=>s!==data.target);
    $('start-state').value=start;
    $('episode-slider').max=episode;$('episode-number').max=episode;$('timeline-end').textContent=episode;
    const successes=data.history.filter(h=>h.success).length;
    $('training-stats').innerHTML=`<span><strong>${data.history.length}</strong> épisodes</span><span><strong>${fmt(100*successes/data.history.length,1)} %</strong> arrivés en G</span>`;
    renderCharts();render();
  }
  function getEvaluation(){
    let current=start, score=0, success=current===data.target, firstLoop=null;
    const path=[current], seen=new Map([[current,0]]), q=snap();
    for(let step=0;step<data.config.max_steps&&!success;step++){
      const stateIndex=data.locations.indexOf(current), action=q.policy[stateIndex];
      const transition=data.transitions[stateIndex][action];
      current=transition.next;score+=transition.reward;path.push(current);success=current===data.target;
      if(!success&&seen.has(current)&&firstLoop===null)firstLoop={from:seen.get(current),to:path.length-1};
      if(!seen.has(current))seen.set(current,path.length-1);
    }
    return {path,score,success,steps:path.length-1,firstLoop};
  }
  function arrow(x,y,action){
    const vectors=[[0,-1],[0,1],[-1,0],[1,0]], [dx,dy]=vectors[action];
    const inner=mapCompact?19:25,outer=mapCompact?31:45;
    const sx=x+dx*inner,sy=y+dy*inner,tx=x+dx*outer,ty=y+dy*outer;
    return `<path class="policy-arrow" d="M ${sx} ${sy} L ${tx} ${ty} M ${tx-dx*6-dy*5} ${ty-dy*6+dx*5} L ${tx} ${ty} L ${tx-dx*6+dy*5} ${ty-dy*6-dx*5}"/>`;
  }
  function renderMap(){
    mapWidth=Math.max(1,$('map').getBoundingClientRect().width);mapCompact=mapWidth<400;
    const mapHeight=mapCompact?350:400;
    $('map').setAttribute('viewBox',`0 0 ${mapWidth} ${mapHeight}`);$('map').style.height=`${mapHeight}px`;
    const q=snap();let html='';const drawn=new Set();
    data.transitions.forEach((row,i)=>row.forEach(t=>{
      const j=data.locations.indexOf(t.next);if(!t.valid||i===j||j<0)return;
      const key=[i,j].sort((a,b)=>a-b).join('-');if(drawn.has(key))return;drawn.add(key);
      const [x1,y1]=point(i),[x2,y2]=point(j);
      html+=`<line class="corridor" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/><line class="corridor-core" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
    }));
    const displayPath=evaluation.firstLoop?evaluation.path.slice(0,evaluation.firstLoop.to+1):evaluation.path;
    html+=`<polyline class="route-line" points="${displayPath.map(s=>point(data.locations.indexOf(s)).join(',')).join(' ')}"/>`;
    data.locations.forEach((state,i)=>{
      const [x,y]=point(i),terminal=state===data.target, max=Math.max(...q.q[i]);
      html+=`<g><circle class="room ${state===start?'start':''} ${terminal?'target':''}" cx="${x}" cy="${y}" r="${mapCompact?18:24}"/><text class="room-label" text-anchor="middle" x="${x}" y="${y+7}">${esc(state)}</text>`;
      if(!terminal)html+=arrow(x,y,q.policy[i]);
      html+=`<text class="room-value" text-anchor="middle" x="${x}" y="${y+(mapCompact?52:64)}">${terminal?'Terminal':fmt(max)}</text></g>`;
    });
    const [rx,ry]=robotPoint(data.locations.indexOf(start));
    html+=`<g id="robot" transform="translate(${rx},${ry})"><circle r="9" fill="var(--purple)" stroke="var(--surface)" stroke-width="2"/><circle cx="-2.6" cy="-1" r="1.2" fill="var(--surface)"/><circle cx="2.6" cy="-1" r="1.2" fill="var(--surface)"/><path d="M -2 3 L 2 3" stroke="var(--surface)" stroke-width="1.2" stroke-linecap="round"/></g>`;
    $('map').innerHTML=html;
  }
  function renderTable(){
    const q=snap();let html='';const all=q.q.flat(), bound=Math.max(1,...all.map(Math.abs));
    data.locations.forEach((state,i)=>{
      const terminal=state===data.target;
      html+=`<tr><th scope="row">${esc(state)}${terminal?'<span class="terminal-label">terminal</span>':''}</th>`;
      q.q[i].forEach((value,j)=>{
        const chosen=!terminal&&q.policy[i]===j, selected=selection[0]===i&&selection[1]===j;
        const color=value<0?'var(--danger)':'var(--blue)', intensity=Math.round(4+Math.abs(value)/bound*17);
        html+=`<td><button type="button" data-state="${esc(state)}" data-action="${esc(data.actions[j])}" class="${chosen?'greedy ':''}${selected?'selected':''}" style="background:color-mix(in srgb, ${color} ${intensity}%, var(--surface))" aria-pressed="${selected}" title="Q(${esc(state)}, ${esc(data.actions[j])}) = ${value}${chosen?' · action gloutonne':''}">${chosen?'★ ':''}${fmt(value,1)}</button></td>`;
      });html+='</tr>';
    });
    $('q-table').querySelector('tbody').innerHTML=html;
    const [i,j]=selection,state=data.locations[i];
    $('q-label').textContent=`Q(${state}, ${data.actions[j]})`;$('q-value').textContent=fmt(q.q[i][j],4);
    $('policy-detail').textContent=state===data.target?'G est terminal : aucune action à choisir, aucune valeur future ajoutée à la cible.':`Depuis ${state}, la politique choisit ${glyphs[q.policy[i]]} ${data.actions[q.policy[i]]}.`;
  }
  function render(){
    if(!data)return;
    $('episode-slider').value=episode;$('episode-number').value=episode;
    const h=data.history[episode-1];
    $('episode-detail').textContent=episode===0?'Avant le premier épisode : toutes les valeurs sont nulles. L’égalité conduit à choisir up.':`Épisode ${episode} · départ ${h.start} · ${h.success?'G atteint':'limite de pas atteinte'} · score ${h.score} · ${h.steps} déplacement${h.steps>1?'s':''}`;
    evaluation=getEvaluation();renderMap();renderTable();
    const display=evaluation.firstLoop?evaluation.path.slice(0,evaluation.firstLoop.to+1):evaluation.path;
    $('route-text').textContent=display.join(' → ')+(evaluation.firstLoop?' → … (boucle)':'');
    $('route-status').textContent=evaluation.success?'G atteint':'G non atteint';$('route-status').classList.toggle('failed',!evaluation.success);
    $('evaluation-summary').textContent=`Score : ${evaluation.score} · ${evaluation.steps} déplacement${evaluation.steps>1?'s':''}${evaluation.success?'':' · limite de '+data.config.max_steps+' pas atteinte'}${evaluation.firstLoop?' · parcours abrégé ; le score inclut tous les pas.':''}`;
    updateChartMarkers();
  }
  function chart(id,series,reference){
    const width=Math.max(1,$(id).getBoundingClientRect().width),height=240,left=57,right=18,top=18,bottom=40,innerW=width-left-right,innerH=height-top-bottom;
    $(id).setAttribute('viewBox',`0 0 ${width} ${height}`);$(id).style.height=`${height}px`;
    chartMetrics.set(id,{left,innerW});
    const count=data.snapshots.length-1,vals=series.flatMap(s=>s.values.map(p=>p[1]));if(reference!==null)vals.push(reference);
    let min=Math.min(0,...vals),max=Math.max(0,...vals);if(max-min<1){max=min+1;}const padding=(max-min)*.08;min-=padding;max+=padding;
    const x=v=>left+innerW*v/Math.max(1,count),y=v=>top+innerH*(1-(v-min)/(max-min));
    let html='';for(let k=0;k<4;k++){const val=min+(max-min)*k/3,yy=y(val);html+=`<line x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}" stroke="var(--line)"/><text x="${left-8}" y="${yy+4}" text-anchor="end" fill="var(--muted)" font-size="11">${fmt(val,0)}</text>`;}
    const ticks=[0,Math.round(count/2),count].filter((v,i,a)=>a.indexOf(v)===i);
    ticks.forEach(v=>html+=`<text x="${x(v)}" y="${height-14}" text-anchor="middle" fill="var(--muted)" font-size="11">${v}</text>`);
    html+=`<text x="${width-right}" y="${height-1}" text-anchor="end" fill="var(--muted)" font-size="11">Épisode</text>`;
    if(reference!==null)html+=`<line x1="${left}" y1="${y(reference)}" x2="${width-right}" y2="${y(reference)}" stroke="var(--green)" stroke-width="1.5" stroke-dasharray="6 5"/>`;
    series.forEach(s=>{const d=s.values.map((p,i)=>`${i?'L':'M'}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join(' ');html+=`<path d="${d}" fill="none" stroke="${s.color||'var(--blue)'}" stroke-width="${s.thick||2}" opacity="${s.opacity||1}" stroke-linecap="round" stroke-linejoin="round"/>`;if(s.values.length===1){html+=`<circle cx="${x(s.values[0][0])}" cy="${y(s.values[0][1])}" r="3" fill="var(--blue)"/>`;}});
    html+=`<line class="episode-marker" x1="${x(episode)}" x2="${x(episode)}" y1="${top}" y2="${height-bottom}" stroke="var(--purple)" stroke-width="1.5" stroke-dasharray="3 3"/>`;
    $(id).innerHTML=html;
  }
  function renderQChart(){
    const [i,j]=selection,state=data.locations[i],action=data.actions[j],reference=state==='A'&&action==='right'?-1-data.config.gamma+100*data.config.gamma**2:null;
    chart('q-chart',[{values:data.snapshots.map((s,k)=>[k,s.q[i][j]])}],reference);
    $('q-chart-title').textContent=`Évolution de Q(${state}, ${action})`;$('reference-key').hidden=reference===null;
    $('q-chart-note').textContent=reference===null?'Le trait vertical repère l’épisode affiché.':`Trajet A → B → C → G : −1 − γ + 100γ² = ${fmt(reference)} avec γ = ${fmt(data.config.gamma)}.`;
  }
  function renderCharts(){
    const raw=data.history.map((h,i)=>[i+1,h.score]);let sum=0;
    const smoothed=data.history.map((h,i)=>{sum+=h.score;if(i>=20)sum-=data.history[i-20].score;return[i+1,sum/Math.min(i+1,20)];});
    chart('score-chart',[{values:raw,opacity:.25,thick:1},{values:smoothed,thick:2.5}],null);renderQChart();
  }
  function updateChartMarkers(){for(const [id,{left,innerW}] of chartMetrics){const marker=$(id).querySelector('.episode-marker'),x=left+innerW*episode/Math.max(1,data.snapshots.length-1);if(marker){marker.setAttribute('x1',x);marker.setAttribute('x2',x);}}}
  function setEpisode(value,fromTimer=false){if(!data||busy)return;if(!fromTimer)stopEpisodes();stopRoute();episode=Math.max(0,Math.min(data.snapshots.length-1,Math.round(Number(value)||0)));render();}
  $('episode-slider').addEventListener('input',e=>setEpisode(e.target.value));
  $('episode-number').addEventListener('change',e=>setEpisode(e.target.value));
  $('start-state').addEventListener('change',e=>{stopAll();start=e.target.value;render();});
  $('q-table').addEventListener('click',e=>{const button=e.target.closest('button[data-state]');if(!button||busy)return;selection=[data.locations.indexOf(button.dataset.state),data.actions.indexOf(button.dataset.action)];renderTable();renderQChart();});
  $('play-episodes').addEventListener('click',()=>{
    if(!data||busy)return;if(episodeTimer){stopEpisodes();return;}stopRoute();if(episode>=data.snapshots.length-1)setEpisode(0);
    $('play-episodes').textContent='Ⅱ Pause';episodeTimer=setInterval(()=>{if(episode>=data.snapshots.length-1){stopEpisodes();return;}setEpisode(episode+1,true);},180);
  });
  $('play-route').addEventListener('click',()=>{
    if(!data||busy)return;if(routeTimer){stopRoute();renderMap();return;}stopEpisodes();routeStep=0;$('play-route').textContent='■ Arrêter';
    const lastAnimatedStep=evaluation.firstLoop?evaluation.firstLoop.to:evaluation.steps;
    const advance=()=>{
      const state=evaluation.path[routeStep],[x,y]=robotPoint(data.locations.indexOf(state));
      $('robot').setAttribute('transform',`translate(${x},${y})`);
      const ended=routeStep>=lastAnimatedStep;
      const detail=ended?(evaluation.firstLoop?' · boucle détectée, animation arrêtée':evaluation.success?' · objectif atteint':' · limite atteinte'):'';
      $('route-progress').textContent=`Pas ${routeStep} / ${evaluation.steps} · état ${state}${detail}`;
      if(ended){clearInterval(routeTimer);routeTimer=null;$('play-route').textContent='↻ Rejouer';}else routeStep++;
    };
    routeTimer=setInterval(advance,400);advance();
  });
  form.addEventListener('submit',async e=>{
    e.preventDefault();if(busy||!form.reportValidity())return;stopAll();showError('');setBusy(true);
    const config=Object.fromEntries(new FormData(form));
    // Disabled controls are omitted by FormData, so read the known numeric fields directly.
    for(const name of ['episodes','alpha','gamma','epsilon','max_steps','seed'])config[name]=Number(form.elements[name].value);
    try{installData(await request('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)}));}catch(error){showError(error.message||String(error));}finally{setBusy(false);}
  });
  async function initialize(){setBusy(true);try{installData(await request('/api/data'));}catch(error){showError(error.message+' Lance ce fichier depuis le script Python pour accéder au serveur local.');}finally{setBusy(false);}}
  let resizeFrame=null;
  const lastWidths=new Map();
  const resizeObserver=new ResizeObserver(entries=>{
    let widthChanged=false;
    for(const entry of entries){const width=entry.contentRect.width;if(Math.abs(width-(lastWidths.get(entry.target)||0))>.5){lastWidths.set(entry.target,width);widthChanged=true;}}
    if(!data||!widthChanged)return;
    cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{stopRoute();renderMap();renderCharts();});
  });
  ['map','score-chart','q-chart'].forEach(id=>resizeObserver.observe($(id)));
  initialize();
})();
</script>
</body>
</html>

"""


if __name__ == "__main__":
    main()
