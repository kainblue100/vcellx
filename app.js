/* ============================================================
   vcellx 原理说明 · 交互部分

   真实计算的三处：
     graphField  受约束换边（度精确保持）与一跳邻域
     maskField   掩码逻辑
     mmdField    能量距离核的 MMD²
   其余为示意绘制。
   ============================================================ */
(function () {
  'use strict';

  var REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var C = {
    graph: '#8b7cf0',
    graphDim: '#544a9e',
    shuffle: '#7d8798',
    direct: '#e9a13c',
    ink: '#0c0f16',
    ink900: '#080a0f',
    grid: '#181d28',
    line: '#303a4a',
    text: '#eef1f7',
    mut: '#778399'
  };

  var REDRAWS = [];
  function redrawOnResize(fn) { REDRAWS.push(fn); }
  (function wire() {
    var t;
    function all() {
      REDRAWS.forEach(function (f) { try { f(); } catch (e) { /* keep going */ } });
    }
    window.addEventListener('resize', function () { clearTimeout(t); t = setTimeout(all, 160); });
    window.addEventListener('load', function () { setTimeout(all, 40); });
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () { setTimeout(all, 40); });
    }
  })();

  function setup(cv) {
    if (!cv) return null;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var r = cv.getBoundingClientRect();
    var w = Math.max(r.width, 240), h = Math.max(r.height, 120);
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    var g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { g: g, w: w, h: h };
  }
  function hexA(hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
  }
  function rng(seed) {
    var s = seed >>> 0;
    return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }

  /* ============================================================
     1. 页首网络背景
     ============================================================ */
  (function heroNet() {
    var cv = document.getElementById('heroNet');
    if (!cv) return;
    var ctx, W, H, nodes = [], edges = [], r = rng(11);

    function build() {
      var s = setup(cv); if (!s) return;
      ctx = s.g; W = s.w; H = s.h;
      nodes = []; edges = [];
      var n = Math.max(26, Math.min(54, Math.round(W / 26)));
      for (var i = 0; i < n; i++) {
        nodes.push({ x: r() * W, y: r() * H, vx: (r() - .5) * .22, vy: (r() - .5) * .22, hub: r() < .16 });
      }
      for (var k = 0; k < n * 1.7; k++) {
        var a = Math.floor(r() * n), b = Math.floor(r() * n);
        if (a !== b) edges.push([a, b]);
      }
    }
    function frame() {
      ctx.clearRect(0, 0, W, H);
      for (var i = 0; i < nodes.length; i++) {
        var p = nodes[i];
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > W) p.vx *= -1;
        if (p.y < 0 || p.y > H) p.vy *= -1;
      }
      ctx.lineWidth = 1;
      edges.forEach(function (e) {
        var a = nodes[e[0]], b = nodes[e[1]];
        ctx.strokeStyle = hexA(C.graph, .11);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      });
      nodes.forEach(function (p) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.hub ? 2.6 : 1.5, 0, 6.2832);
        ctx.fillStyle = p.hub ? hexA(C.graph, .55) : hexA(C.graph, .26);
        ctx.fill();
      });
      if (!REDUCED) requestAnimationFrame(frame);
    }
    build(); frame();
    redrawOnResize(function () { build(); if (REDUCED) frame(); });
  })();

  /* ============================================================
     2. 图 1 · 一跳邻域 + 度匹配打乱
     ============================================================ */
  (function graphField() {
    var cv = document.getElementById('graphField');
    if (!cv) return;
    var btnReal = document.getElementById('topoReal');
    var btnShuf = document.getElementById('topoShuf');
    var sNeighbor = document.getElementById('gNeighbor');
    var vNeighbor = document.getElementById('gNeighborVal');
    var outTarget = document.getElementById('gTarget');
    var outDeg = document.getElementById('gDeg');
    var outEdges = document.getElementById('gEdges');

    var names = ['ACAT2', 'AKT2', 'ANTXR1', 'ARID1A', 'CLDN6', 'DHCR24', 'DHX36', 'DNAJA3',
                 'EP300', 'FGFR1', 'GATA3', 'HDAC2', 'IKBKB', 'JAK2', 'KDM5A', 'LRP6',
                 'MAP2K1', 'MYC', 'NF1', 'OGG1', 'PIK3CA', 'RB1', 'SMAD4', 'TSC1'];

    var n = 16, real = [], shuf = [], shuffleMode = false, target = 0, pos = [];

    function buildTopology(seed) {
      var r = rng(seed);
      var E = [];
      /* enough edges that the target has a real neighbourhood: ~5 neighbours
         at n=16, which is what makes the diffusion story legible */
      var k = Math.round(n * 2.6);
      for (var i = 0; i < k; i++) {
        var a = Math.floor(r() * n), b = Math.floor(r() * n);
        if (a === b) continue;
        E.push([Math.min(a, b), Math.max(a, b), 0.25 + r() * 0.75]);
      }
      // dedupe
      var seen = {}, out = [];
      E.forEach(function (e) {
        var key = e[0] + '-' + e[1];
        if (seen[key]) return;
        seen[key] = 1; out.push(e);
      });
      return out;
    }

    /* 受约束换边：交换两条边共享的端点，保持每个节点度精确不变 */
    function degreeMatchedShuffle(E, nNodes, seed, rounds) {
      var r = rng(seed);
      var edges = E.map(function (e) { return [e[0], e[1], e[2]]; });
      var adj = [];
      for (var i = 0; i < nNodes; i++) adj.push({});

      function rebuild() {
        adj = [];
        for (var i = 0; i < nNodes; i++) adj.push({});
        edges.forEach(function (e) {
          adj[e[0]][e[1]] = 1;
          adj[e[1]][e[0]] = 1;
        });
      }
      rebuild();

      var done = 0, tries = 0, want = rounds;
      while (done < want && tries < want * 60) {
        tries++;
        var i1 = Math.floor(r() * edges.length), i2 = Math.floor(r() * edges.length);
        if (i1 === i2) continue;
        var a = edges[i1][0], b = edges[i1][1];
        var c = edges[i2][0], d = edges[i2][1];
        // 尝试把 (a,b),(c,d) 换成 (a,d),(c,b)
        if (a === d || c === b) continue;
        if (adj[a][d] || adj[c][b]) continue;     // 会产生重复边
        delete adj[a][b]; delete adj[b][a];
        delete adj[c][d]; delete adj[d][c];
        adj[a][d] = 1; adj[d][a] = 1;
        adj[c][b] = 1; adj[b][c] = 1;
        edges[i1] = [a, d, edges[i1][2]];
        edges[i2] = [c, b, edges[i2][2]];
        done++;
      }
      return edges;
    }

    function rebuild() {
      n = +sNeighbor.value;
      vNeighbor.textContent = String(n);
      real = buildTopology(20261116);
      shuf = degreeMatchedShuffle(real, n, 777, real.length * 8);
      if (target >= n) target = 0;
      // 环形布局 + 轻微抖动
      var r = rng(5);
      pos = [];
      for (var i = 0; i < n; i++) {
        var a = (i / n) * Math.PI * 2 - Math.PI / 2;
        pos.push({ a: a, jr: (r() - .5) * 0.12 });
      }
      draw();
    }

    function neighborsOf(E) {
      var nb = {};
      E.forEach(function (e) {
        if (e[0] === target) nb[e[1]] = e[2];
        if (e[1] === target) nb[e[0]] = e[2];
      });
      return nb;
    }

    function draw() {
      var s = setup(cv); if (!s) return;
      var g = s.g, W = s.w, H = s.h;
      g.clearRect(0, 0, W, H);
      var E = shuffleMode ? shuf : real;
      var nb = neighborsOf(E);
      var deg = Object.keys(nb).length;

      var cx = W / 2, cy = H / 2 + 6;
      var R = Math.min(W, H) * 0.34;

      function xy(i) {
        var a = pos[i].a;
        var rr = R * (1 + pos[i].jr);
        return { x: cx + Math.cos(a) * rr, y: cy + Math.sin(a) * rr };
      }

      // 边
      var maxW = 1e-6;
      E.forEach(function (e) { if (e[2] > maxW) maxW = e[2]; });
      E.forEach(function (e) {
        var touches = (e[0] === target || e[1] === target);
        var a = xy(e[0]), b = xy(e[1]);
        g.strokeStyle = touches ? hexA(C.graph, .70) : hexA(C.text, .13);
        g.lineWidth = touches ? 1.2 + (e[2] / maxW) * 2.6 : 1.1;
        g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
      });

      // 节点
      for (var i = 0; i < n; i++) {
        var p = xy(i);
        var isTarget = (i === target);
        var isNb = nb[i] !== undefined;
        var rad = isTarget ? 8.5 : (isNb ? 5.5 : 3.6);
        var col = isTarget ? C.direct : (isNb ? C.graph : C.mut);
        var alpha = isTarget ? 1 : (isNb ? .92 : .42);
        g.beginPath(); g.arc(p.x, p.y, rad, 0, 6.2832);
        g.fillStyle = hexA(col, alpha); g.fill();
        if (isTarget) {
          g.strokeStyle = C.text; g.lineWidth = 1.6;
          g.beginPath(); g.arc(p.x, p.y, rad + 3.5, 0, 6.2832); g.stroke();
        }
        if (isTarget || isNb) {
          g.font = '10px "IBM Plex Mono", monospace';
          g.textAlign = 'center';
          /* outline first so a thick edge crossing the label does not swallow it */
          g.lineWidth = 3.5;
          g.strokeStyle = C.ink900;
          g.strokeText(names[i % names.length], p.x, p.y - rad - 6);
          g.fillStyle = isTarget ? C.direct : hexA(C.text, .82);
          g.fillText(names[i % names.length], p.x, p.y - rad - 6);
          g.textAlign = 'start';
        }
      }

      // 说明
      g.font = '10px "IBM Plex Mono", monospace';
      g.fillStyle = C.mut;
      g.fillText(shuffleMode ? '拓扑：度匹配打乱（度不变，连接改变）' : '拓扑：真实 Hi-C 环', 10, 16);

      // 图例
      var ly = 32;
      [[C.direct, '靶基因'], [C.graph, '一跳邻居'], [C.mut, '其它基因']].forEach(function (it, k) {
        var yy = ly + k * 14;
        g.beginPath(); g.arc(14, yy - 3, 3.4, 0, 6.2832);
        g.fillStyle = hexA(it[0], .95); g.fill();
        g.fillStyle = hexA(C.text, .55);
        g.fillText(it[1], 24, yy);
      });
      g.fillStyle = hexA(C.text, .40);
      g.fillText('边越粗 = Hi-C observed 越高', 10, ly + 3 * 14 + 2);

      g.fillStyle = hexA(C.text, .5);
      g.fillText('点击任意节点切换靶基因', 10, H - 10);

      outTarget.textContent = names[target % names.length];
      outDeg.textContent = String(deg);
      outEdges.textContent = String(E.length);
    }

    cv.addEventListener('click', function (e) {
      var rect = cv.getBoundingClientRect();
      var mx = e.clientX - rect.left, my = e.clientY - rect.top;
      var cx = rect.width / 2, cy = rect.height / 2 + 6;
      var R = Math.min(rect.width, rect.height) * 0.34;
      var best = -1, bestD = 1e9;
      for (var i = 0; i < n; i++) {
        var a = pos[i].a, rr = R * (1 + pos[i].jr);
        var px = cx + Math.cos(a) * rr, py = cy + Math.sin(a) * rr;
        var d = (px - mx) * (px - mx) + (py - my) * (py - my);
        if (d < bestD) { bestD = d; best = i; }
      }
      if (best >= 0 && bestD < 900) { target = best; draw(); }
    });

    function setMode(isShuffle) {
      shuffleMode = isShuffle;
      btnReal.setAttribute('aria-pressed', String(!isShuffle));
      btnShuf.setAttribute('aria-pressed', String(isShuffle));
      draw();
    }
    btnReal.addEventListener('click', function () { setMode(false); });
    btnShuf.addEventListener('click', function () { setMode(true); });
    sNeighbor.addEventListener('input', rebuild);

    rebuild();
    redrawOnResize(draw);
  })();

  /* ============================================================
     3. 图 2 · 掩码与重建
     ============================================================ */
  (function maskField() {
    var cv = document.getElementById('maskField');
    if (!cv) return;
    var sFrac = document.getElementById('maskFrac');
    var vFrac = document.getElementById('maskFracVal');
    var btn = document.getElementById('maskResample');
    var outCount = document.getElementById('maskCount');

    /* Cells must read as squares, not bars. On a 16:9 canvas that means the
       column count has to be about 16/9 x the row count, otherwise each cell is
       a tall rectangle and the figure looks like a bar chart. */
    var nRows = 12, NCOL = 34, mask = [], vals = [];

    function resample() {
      var frac = +sFrac.value / 100;
      vFrac.textContent = frac.toFixed(2);
      outCount.textContent = Math.round(4096 * frac).toLocaleString('en-US');
      var r = rng(Date.now() & 0xffff);
      mask = []; vals = [];
      for (var i = 0; i < NCOL * nRows; i++) {
        mask.push(r() < frac);
        vals.push(r());
      }
      draw();
    }

    function draw() {
      var s = setup(cv); if (!s) return;
      var g = s.g, W = s.w, H = s.h;
      g.clearRect(0, 0, W, H);

      var padL = 12, padT = 40, padB = 30, padR = 12;
      var cw = (W - padL - padR) / NCOL;
      var ch = (H - padT - padB) / nRows;

      for (var i = 0; i < NCOL * nRows; i++) {
        var col = i % NCOL, row = Math.floor(i / NCOL);
        var x = padL + col * cw, y = padT + row * ch;
        var m = mask[i];
        if (m) {
          g.fillStyle = hexA(C.graph, .38 + vals[i] * 0.55);
        } else {
          /* kept genes must be clearly visible, just less saturated */
          g.fillStyle = hexA(C.text, .16 + vals[i] * 0.20);
        }
        g.fillRect(x + 0.6, y + 0.6, Math.max(cw - 1.2, 1), Math.max(ch - 1.2, 1));
        if (m) {
          g.strokeStyle = hexA(C.graph, .75);
          g.lineWidth = 0.8;
          g.strokeRect(x + 0.6, y + 0.6, Math.max(cw - 1.2, 1), Math.max(ch - 1.2, 1));
        }
      }

      g.font = '11px "IBM Plex Mono", monospace';
      g.fillStyle = hexA(C.text, .78);
      g.fillText('未遮住，编码器看得到', padL, 18);
      g.fillStyle = C.graph;
      g.fillText('被遮住，只从 256 维向量重建', padL + W * 0.44, 18);
      g.fillStyle = hexA(C.text, .45);
      g.fillText('4,096 个面板基因 → 单个向量 → 只重建紫色位置', padL, H - 8);
    }

    sFrac.addEventListener('input', resample);
    btn.addEventListener('click', resample);
    resample();
    redrawOnResize(draw);
  })();

  /* ============================================================
     4. 图 3 · MMD
     ============================================================ */
  (function mmd() {
    var cv = document.getElementById('mmdField');
    if (!cv) return;
    var sShift = document.getElementById('mShift');
    var sSpread = document.getElementById('mSpread');
    var vShift = document.getElementById('mShiftVal');
    var vSpread = document.getElementById('mSpreadVal');
    var outVal = document.getElementById('mVal');
    var outBest = document.getElementById('mBest');
    var btnFit = document.getElementById('mFit');

    var NP = 200, r = rng(20261116);
    var CX = 0.32, CY = 0.52, base = [];
    for (var i = 0; i < NP; i++) {
      var a = (i + .5) / NP * Math.PI * 2;
      var rad = Math.sqrt((i + .5) / NP);
      base.push({
        dx: Math.cos(a) * rad * 0.145 + (r() - .5) * 0.035,
        dy: Math.sin(a) * rad * 0.112 + (r() - .5) * 0.035
      });
    }
    function predicted(shift, spread) {
      return base.map(function (p) {
        return { x: CX + shift * 0.30 + p.dx * spread, y: CY + p.dy * spread };
      });
    }
    /* 观测分布：右移 + 纵向拉伸 + 剪切。预测的两个自由度拟合不了剪切。 */
    var OBS = base.map(function (p) {
      return { x: 0.462 + p.dx * 0.94 + p.dy * 0.60, y: CY + p.dy * 1.12 };
    });

    function meanPairwise(A, B) {
      var s = 0;
      for (var i = 0; i < A.length; i++) {
        for (var j = 0; j < B.length; j++) {
          var dx = A[i].x - B[j].x, dy = A[i].y - B[j].y;
          s += Math.sqrt(dx * dx + dy * dy);
        }
      }
      return s / (A.length * B.length);
    }
    var OO = meanPairwise(OBS, OBS);
    function mmdValue(P) {
      /* 能量距离 D = 2E||x-y|| - E||x-x'|| - E||y-y'|| */
      return Math.max(0, 2 * meanPairwise(P, OBS) - meanPairwise(P, P) - OO);
    }

    var curve = [];
    function computeCurve(spread) {
      curve = [];
      for (var k = 0; k <= 56; k++) {
        var sh = k / 56 * 1.4;
        curve.push({ s: sh, v: mmdValue(predicted(sh, spread)) });
      }
    }

    function draw() {
      var s = setup(cv); if (!s) return;
      var g = s.g, W = s.w, H = s.h;
      g.clearRect(0, 0, W, H);

      var shift = +sShift.value / 100, spread = +sSpread.value / 100;
      var PRED = predicted(shift, spread);

      g.strokeStyle = C.grid; g.lineWidth = 1; g.globalAlpha = .5;
      for (var i = 1; i < 8; i++) {
        g.beginPath(); g.moveTo(W * i / 8, 0); g.lineTo(W * i / 8, H); g.stroke();
        g.beginPath(); g.moveTo(0, H * i / 8); g.lineTo(W, H * i / 8); g.stroke();
      }
      g.globalAlpha = 1;

      function dots(A, fill, rr) {
        g.fillStyle = fill;
        for (var i = 0; i < A.length; i++) {
          g.beginPath(); g.arc(A[i].x * W, A[i].y * H, rr, 0, 6.2832); g.fill();
        }
      }
      dots(OBS, 'rgba(200,210,225,.30)', 2.2);
      dots(PRED, hexA(C.graph, .62), 2.4);

      g.font = '11px "IBM Plex Mono", monospace';
      g.fillStyle = '#c8d2e1'; g.fillText('观测', 10, 18);
      g.fillStyle = C.graph; g.fillText('预测', 50, 18);

      drawCurve(g, W, H, shift, spread);
    }

    function drawCurve(g, W, H, shift, spread) {
      var cw = Math.min(W * 0.32, 230), ch = Math.min(H * 0.28, 92);
      var ox = W - cw - 16, oy = H - ch - 16;
      g.fillStyle = 'rgba(8,10,15,.88)';
      g.fillRect(ox - 9, oy - 19, cw + 18, ch + 30);
      g.strokeStyle = C.line; g.lineWidth = 1;
      g.strokeRect(ox - 9, oy - 19, cw + 18, ch + 30);

      g.font = '10px "IBM Plex Mono", monospace';
      g.fillStyle = C.mut;
      g.fillText('MMD² 随位移', ox, oy - 5);

      var maxV = 0;
      curve.forEach(function (p) { if (p.v > maxV) maxV = p.v; });
      maxV = maxV || 1;

      g.strokeStyle = hexA(C.graph, .9); g.lineWidth = 2;
      g.beginPath();
      curve.forEach(function (p, i) {
        var px = ox + (p.s / 1.4) * cw;
        var py = oy + ch - (p.v / maxV) * ch * 0.76 - ch * 0.10;
        i ? g.lineTo(px, py) : g.moveTo(px, py);
      });
      g.stroke();
      g.lineTo(ox + cw, oy + ch); g.lineTo(ox, oy + ch); g.closePath();
      g.fillStyle = hexA(C.graph, .13); g.fill();
      g.strokeStyle = hexA(C.text, .2); g.lineWidth = 1;
      g.beginPath(); g.moveTo(ox, oy + ch); g.lineTo(ox + cw, oy + ch); g.stroke();

      var best = curve.reduce(function (a, b) { return b.v < a.v ? b : a; }, curve[0]);
      g.fillStyle = '#fde725';
      g.beginPath();
      g.arc(ox + (best.s / 1.4) * cw, oy + ch - (best.v / maxV) * ch * 0.76 - ch * 0.10, 3, 0, 6.2832);
      g.fill();

      var cur = mmdValue(predicted(shift, spread));
      var cxp = ox + (shift / 1.4) * cw;
      var cyp = oy + ch - (cur / maxV) * ch * 0.76 - ch * 0.10;
      g.strokeStyle = C.direct; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(cxp, oy - 2); g.lineTo(cxp, oy + ch); g.stroke();
      g.fillStyle = C.direct;
      g.beginPath(); g.arc(cxp, cyp, 4, 0, 6.2832); g.fill();
      g.strokeStyle = C.ink; g.lineWidth = 1.4;
      g.beginPath(); g.arc(cxp, cyp, 4, 0, 6.2832); g.stroke();
    }

    function sync() {
      var shift = +sShift.value / 100, spread = +sSpread.value / 100;
      vShift.textContent = shift.toFixed(2);
      vSpread.textContent = spread.toFixed(2);
      computeCurve(spread);
      var best = curve.reduce(function (a, b) { return b.v < a.v ? b : a; }, curve[0]);
      var cur = mmdValue(predicted(shift, spread));
      outVal.textContent = cur.toFixed(3);
      outVal.style.color = (cur - best.v) < 0.0015 ? 'var(--ok)' : 'var(--graph)';
      outBest.textContent = best.s.toFixed(2);
      draw();
    }

    sShift.addEventListener('input', sync);
    sSpread.addEventListener('input', sync);

    btnFit.addEventListener('click', function () {
      var bestS = 0, bestSp = 1, bestV = Infinity;
      for (var sp = 0.55; sp <= 1.55; sp += 0.05) {
        for (var k = 0; k <= 40; k++) {
          var sh = k / 40 * 1.4;
          var v = mmdValue(predicted(sh, sp));
          if (v < bestV) { bestV = v; bestS = sh; bestSp = sp; }
        }
      }
      var tS = Math.round(bestS * 100), tP = Math.round(bestSp * 100);
      if (REDUCED) { sShift.value = tS; sSpread.value = tP; sync(); return; }
      var fS = +sShift.value, fP = +sSpread.value, t0 = performance.now(), dur = 460;
      (function step(now) {
        var k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
        sShift.value = Math.round(fS + (tS - fS) * e);
        sSpread.value = Math.round(fP + (tP - fP) * e);
        sync();
        if (k < 1) requestAnimationFrame(step);
      })(t0);
    });

    sync();
    redrawOnResize(sync);
  })();

  /* ============================================================
     滚动入场
     ============================================================ */
  (function reveal() {
    var nodes = document.querySelectorAll('[data-reveal]');
    if (REDUCED || !('IntersectionObserver' in window)) {
      Array.prototype.forEach.call(nodes, function (n) { n.classList.add('is-in'); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e, i) {
        if (!e.isIntersecting) return;
        var el = e.target;
        el.style.transitionDelay = (Math.min(i, 4) * 60) + 'ms';
        el.classList.add('is-in');
        io.unobserve(el);
      });
    }, { rootMargin: '0px 0px -10% 0px', threshold: 0.04 });
    Array.prototype.forEach.call(nodes, function (n) { io.observe(n); });
  })();

})();
