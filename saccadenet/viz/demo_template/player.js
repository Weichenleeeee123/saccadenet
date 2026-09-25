(function () {
  "use strict";
  const D = window.DEMO_DATA;
  const $ = (id) => document.getElementById(id);
  const canvas = $("stage");
  const ctx = canvas.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  const C = (name) => css.getPropertyValue(name).trim();
  const METHODS = [
    ["saccadenet_lite", "SaccadeNet"],
    ["two_stage", "两阶段"],
    ["downsample_1stage", "一段缩图"],
    ["full_res_sliding", "全分辨率滑窗"],
  ];
  const COLORS = { saccadenet_lite: C("--gaze"), two_stage: "#e05d78", downsample_1stage: "#8a7fe0", full_res_sliding: "#e8a07f" };
  const state = { ep: 0, step: 0, view: "seen", playing: false, timer: null };
  const cache = new Map();

  function img(src) {
    if (!cache.has(src)) {
      const image = new Image();
      image.src = src;
      cache.set(src, image);
    }
    return cache.get(src);
  }
  const pad = (n) => String(n).padStart(3, "0");
  const ep = () => D.episodes[state.ep];
  const hitOf = (e) => (e.baselines.saccadenet_lite ? e.baselines.saccadenet_lite.hit === 1 : null);
  const resLabel = (w) => ({ 1920: "1080p", 3840: "4K", 7680: "8K", 15360: "16K" }[w] || `${w}px`);

  function fmtFlops(flops) {
    if (flops < 1e9) return [(flops / 1e6).toFixed(flops < 1e7 ? 2 : 1), "MFLOPs"];
    const g = flops / 1e9;
    return [g >= 100 ? g.toLocaleString("en-US", { maximumFractionDigits: 0 }) : g.toFixed(2), "GFLOPs"];
  }
  const fmtShort = (flops) => fmtFlops(flops).join(" ").replace("FLOPs", "");

  function preload(e) {
    img(`${e.id}/canvas.jpg`);
    e.steps.forEach((_, i) => {
      img(`${e.id}/seen-${pad(i + 1)}.jpg`);
      img(`${e.id}/zoom-${pad(i + 1)}.jpg`);
      img(`${e.id}/fovea-${pad(i + 1)}.png`);
    });
  }

  function buildTabs() {
    const tabs = $("tabs");
    tabs.innerHTML = "";
    D.episodes.forEach((e, index) => {
      const button = document.createElement("button");
      const hit = hitOf(e);
      button.innerHTML = `<span class="dot" style="background:${hit ? C("--truth") : C("--miss")}"></span>${resLabel(e.width)} · 种子 ${e.seed} · ${hit ? "命中" : "未命中"}`;
      button.onclick = () => select(index);
      tabs.appendChild(button);
    });
  }

  function select(index) {
    stop();
    state.ep = index;
    state.step = 0;
    const e = ep();
    preload(e);
    canvas.width = e.stage[0];
    canvas.height = e.stage[1];
    [...$("tabs").children].forEach((b, i) => b.classList.toggle("on", i === index));
    $("scrub").max = e.steps.length - 1;
    $("stepTotal").textContent = e.steps.length;
    $("subtitle").textContent = `${e.width.toLocaleString()} × ${e.height.toLocaleString()} 画布 · 12 张卡片里找数字「${e.query}」${e.note ? " · " + e.note : ""}`;
    $("zoomSpan").textContent = `${e.zoom_span.toLocaleString()} px 视野`;
    render();
  }

  function drawStage(e, s, last) {
    const bg = img(state.view === "real" ? `${e.id}/canvas.jpg` : `${e.id}/seen-${pad(state.step + 1)}.jpg`);
    const paint = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(bg, 0, 0, canvas.width, canvas.height);
      const k = canvas.width / e.width;
      const P = (xy) => [xy[0] * k, xy[1] * k];
      const unit = canvas.width / 1280;

      // Fixation path so far.
      const path = e.steps.slice(0, state.step + 1).map((t) => P(t.fixation));
      ctx.save();
      ctx.lineWidth = 2.2 * unit;
      ctx.strokeStyle = C("--gaze");
      ctx.shadowColor = "rgba(63,214,204,0.7)";
      ctx.shadowBlur = 8 * unit;
      ctx.beginPath();
      path.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.stroke();
      ctx.restore();
      path.forEach(([x, y], i) => {
        if (i === path.length - 1) return;
        ctx.fillStyle = "rgba(63,214,204,0.85)";
        ctx.beginPath();
        ctx.arc(x, y, 3.2 * unit, 0, Math.PI * 2);
        ctx.fill();
      });

      // Candidates with posterior.
      const ranked = s.candidates.map((c, i) => ({ ...c, p: s.posterior[i] || 0 })).sort((a, b) => b.p - a.p);
      ranked.forEach((c, rank) => {
        const [x, y] = P(c.xy);
        const r = (7 + 12 * c.p) * unit;
        ctx.lineWidth = 2 * unit;
        ctx.strokeStyle = rank === 0 && c.p > 0.5 ? C("--gaze") : C("--cand");
        ctx.fillStyle = rank === 0 && c.p > 0.5 ? "rgba(63,214,204,0.18)" : "rgba(243,169,60,0.12)";
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        if (rank < 3 && c.p >= 0.01) {
          ctx.font = `${12 * unit}px ${C("--mono")}`;
          ctx.lineWidth = 3 * unit;
          ctx.strokeStyle = "rgba(0,0,0,0.8)";
          const label = `#${c.id} ${c.p.toFixed(2)}`;
          ctx.strokeText(label, x + r + 4 * unit, y + 4 * unit);
          ctx.fillStyle = "#fff";
          ctx.fillText(label, x + r + 4 * unit, y + 4 * unit);
        }
      });

      // Current fixation reticle.
      const [fx, fy] = path[path.length - 1];
      ctx.save();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2 * unit;
      ctx.shadowColor = "rgba(63,214,204,0.9)";
      ctx.shadowBlur = 12 * unit;
      ctx.beginPath();
      ctx.arc(fx, fy, 13 * unit, 0, Math.PI * 2);
      ctx.stroke();
      [[-22, 0, -8, 0], [8, 0, 22, 0], [0, -22, 0, -8], [0, 8, 0, 22]].forEach(([a, b, c, d]) => {
        ctx.beginPath();
        ctx.moveTo(fx + a * unit, fy + b * unit);
        ctx.lineTo(fx + c * unit, fy + d * unit);
        ctx.stroke();
      });
      ctx.restore();

      if (last) {
        const hit = hitOf(e);
        const [tx, ty] = P(e.truth);
        ctx.lineWidth = 3 * unit;
        ctx.strokeStyle = C("--truth");
        ctx.beginPath();
        ctx.arc(tx, ty, 22 * unit, 0, Math.PI * 2);
        ctx.stroke();
        if (e.answer) {
          const [ax, ay] = P(e.answer);
          ctx.strokeStyle = hit ? C("--truth") : C("--miss");
          ctx.lineWidth = 3.5 * unit;
          ctx.beginPath();
          ctx.moveTo(ax - 10 * unit, ay - 10 * unit); ctx.lineTo(ax + 10 * unit, ay + 10 * unit);
          ctx.moveTo(ax - 10 * unit, ay + 10 * unit); ctx.lineTo(ax + 10 * unit, ay - 10 * unit);
          ctx.stroke();
        }
      }
    };
    if (bg.complete) paint(); else bg.onload = paint;
  }

  function drawBars(s) {
    const bars = $("bars");
    const ranked = s.candidates.map((c, i) => ({ id: c.id, p: s.posterior[i] || 0 })).sort((a, b) => b.p - a.p || a.id - b.id);
    $("candCount").textContent = `${s.candidates.length} / 12 已发现`;
    if (!ranked.length) {
      bars.innerHTML = '<li class="empty">还没发现候选：按固定网格探索</li>';
      return;
    }
    bars.innerHTML = ranked.map((c, i) => `<li class="${i === 0 ? "top" : ""}"><span>#${c.id}</span><span class="track"><span class="fill" style="width:${(c.p * 100).toFixed(1)}%"></span></span><span>${c.p.toFixed(3)}</span></li>`).join("");
  }

  function drawCompare(e, s) {
    const rows = [];
    METHODS.forEach(([key, name]) => {
      if (key === "saccadenet_lite") rows.push({ key, name, value: s.semantic_flops, note: `第 ${state.step + 1} 眼累计` });
      else if (e.baselines[key]) {
        const b = e.baselines[key];
        rows.push({ key, name, value: b.semantic_flops, note: `${b.hit ? "命中" : "未命中"} · ${b.seconds.toFixed(2)} s` });
      }
    });
    const floor = 1e7;
    const top = Math.max(...rows.map((r) => r.value), ...METHODS.map(([k]) => (e.baselines[k] ? e.baselines[k].semantic_flops : 0)));
    const width = (v) => Math.max(1.5, (100 * (Math.log10(Math.max(v, floor)) - Math.log10(floor))) / (Math.log10(top) - Math.log10(floor)));
    $("compare").innerHTML = rows.map((r) => `<div class="row ${r.key === "saccadenet_lite" ? "me" : ""}"><span class="name">${r.name}</span><span class="track"><span class="fill" style="width:${width(r.value).toFixed(1)}%;background:${COLORS[r.key]}"></span></span><span class="num">${fmtShort(r.value)} · ${r.note}</span></div>`).join("");
  }

  function render() {
    const e = ep();
    const s = e.steps[state.step];
    const last = state.step === e.steps.length - 1;
    drawStage(e, s, last);
    $("stepNo").textContent = state.step + 1;
    $("scrub").value = state.step;
    $("fovea").src = `${e.id}/fovea-${pad(state.step + 1)}.png`;
    $("zoom").src = `${e.id}/zoom-${pad(state.step + 1)}.jpg`;
    const [value, unit] = fmtFlops(s.semantic_flops);
    $("semantic").textContent = value;
    $("semanticUnit").textContent = unit;
    $("bytes").textContent = (s.sensing_bytes / 1e6).toFixed(1);
    $("maxp").textContent = Math.max(0, ...s.posterior).toFixed(3);
    drawBars(s);
    drawCompare(e, s);
    const reasons = { threshold: "最大后验越过 τ=0.95，停止", search_exhausted: "探索锚点用完，给出当前最佳", max_steps: "达到 40 眼上限" };
    $("stageNote").textContent = last
      ? `${reasons[e.reason] || e.reason} · ${hitOf(e) ? "✓ 命中目标" : "✗ 停在错误卡片"}（绿圈=真值，仅评测层叠加）`
      : state.view === "seen"
        ? "网络所见：只用这一眼的视网膜样本重建，越远越糊"
        : "真实画面（网络从不直接读取它）";
  }

  function go(step) {
    const e = ep();
    state.step = Math.max(0, Math.min(e.steps.length - 1, step));
    render();
  }
  function stop() {
    state.playing = false;
    clearInterval(state.timer);
    $("play").textContent = "▶";
  }
  function play() {
    const e = ep();
    if (state.step >= e.steps.length - 1) state.step = 0;
    state.playing = true;
    $("play").textContent = "⏸";
    render();
    state.timer = setInterval(() => {
      if (state.step >= ep().steps.length - 1) return stop();
      go(state.step + 1);
    }, Number($("speed").value));
  }

  $("play").onclick = () => (state.playing ? stop() : play());
  $("prev").onclick = () => { stop(); go(state.step - 1); };
  $("next").onclick = () => { stop(); go(state.step + 1); };
  $("first").onclick = () => { stop(); go(0); };
  $("scrub").oninput = (event) => { stop(); go(Number(event.target.value)); };
  $("speed").onchange = () => { if (state.playing) { stop(); play(); } };
  document.querySelectorAll(".view-toggle button").forEach((button) => {
    button.onclick = () => {
      state.view = button.dataset.view;
      document.querySelectorAll(".view-toggle button").forEach((b) => b.classList.toggle("on", b === button));
      render();
    };
  });
  document.addEventListener("keydown", (event) => {
    if (event.target.tagName === "SELECT") return;
    if (event.key === " ") { event.preventDefault(); state.playing ? stop() : play(); }
    else if (event.key === "ArrowRight") { stop(); go(state.step + 1); }
    else if (event.key === "ArrowLeft") { stop(); go(state.step - 1); }
    else if (event.key === "Home") { stop(); go(0); }
    else if (event.key === "End") { stop(); go(Infinity); }
    else if (event.key.toLowerCase() === "v") document.querySelector(`.view-toggle button[data-view="${state.view === "seen" ? "real" : "seen"}"]`).click();
  });

  $("foot").innerHTML = `来源 <code>${D.source_run}</code> · Git <code>${D.git_head.slice(0, 8)}</code> · 权重 SHA256 <code>${D.checkpoint_sha256.slice(0, 12)}…</code>。逐帧由记录的种子确定性重建画布、按保存的 trace 回放，没有重新推理。“网络所见”只用当前这一眼的视网膜样本重建；语义算量为解析估计（CNN 按 1 MAC = 2 FLOPs）。空格播放，←/→ 逐眼，V 切换画面。`;
  buildTabs();
  const params = new URLSearchParams(location.search);
  select(Math.min(D.episodes.length - 1, Math.max(0, Number(params.get("ep")) || 0)));
  if (params.get("view") === "real") document.querySelector('.view-toggle button[data-view="real"]').click();
  if (params.has("step")) go(Number(params.get("step")) - 1);
})();
