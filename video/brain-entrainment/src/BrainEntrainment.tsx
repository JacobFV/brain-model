import React from "react";
import {
  AbsoluteFill,
  Audio,
  Img,
  Sequence,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  Easing,
} from "remotion";
import scriptData from "../public/script.json";

const FPS = 30;
const segments = scriptData.segments;

// ── Animated Brain SVG ───────────────────────────────────────────────────

const BrainSvg: React.FC<{
  activation: number; // 0-1, how "lit up"
  color: string;
  x: number;
  y: number;
  size: number;
}> = ({ activation, color, x, y, size }) => {
  const frame = useCurrentFrame();
  const pulse = 1 + Math.sin(frame * 0.1) * 0.02 * activation;

  return (
    <g transform={`translate(${x}, ${y}) scale(${pulse})`}>
      {/* Brain outline */}
      <ellipse cx={0} cy={0} rx={size * 0.48} ry={size * 0.4}
        fill="none" stroke={color} strokeWidth={2} opacity={0.3 + activation * 0.7} />
      {/* Left hemisphere */}
      <path
        d={`M 0 ${-size * 0.35} C ${-size * 0.5} ${-size * 0.35}, ${-size * 0.55} ${size * 0.1}, ${-size * 0.3} ${size * 0.35} C ${-size * 0.1} ${size * 0.4}, 0 ${size * 0.2}, 0 0`}
        fill={color} opacity={0.15 + activation * 0.4}
      />
      {/* Right hemisphere */}
      <path
        d={`M 0 ${-size * 0.35} C ${size * 0.5} ${-size * 0.35}, ${size * 0.55} ${size * 0.1}, ${size * 0.3} ${size * 0.35} C ${size * 0.1} ${size * 0.4}, 0 ${size * 0.2}, 0 0`}
        fill={color} opacity={0.15 + activation * 0.4}
      />
      {/* Fissures */}
      <path d={`M 0 ${-size * 0.3} L 0 ${size * 0.3}`}
        stroke={color} strokeWidth={1.5} opacity={0.4 + activation * 0.3} />
      {/* Activation glow */}
      {activation > 0.3 && (
        <circle cx={0} cy={0} r={size * 0.3 * activation}
          fill={color} opacity={activation * 0.15} />
      )}
    </g>
  );
};

// ── Entrainment Funnel Animation ─────────────────────────────────────────

const EntrainmentFunnel: React.FC<{
  progress: number; // 0 = scattered, 1 = converged
}> = ({ progress }) => {
  const frame = useCurrentFrame();

  const states = [
    { label: "Anxious", color: "#FF6B6B", baseX: -180, baseY: -200 },
    { label: "Angry", color: "#FF4444", baseX: 180, baseY: -150 },
    { label: "Excited", color: "#FFD93D", baseX: -160, baseY: 100 },
    { label: "Focused", color: "#4ECDC4", baseX: 200, baseY: 150 },
    { label: "Resting", color: "#95E1D3", baseX: 0, baseY: 250 },
  ];

  // Center point
  const centerX = 540;
  const centerY = 700;

  return (
    <svg viewBox="0 0 1080 1400" style={{ width: "100%", height: "auto", position: "absolute", top: 100 }}>
      {/* Funnel shape */}
      <defs>
        <linearGradient id="funnelGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(100,100,255,0.05)" />
          <stop offset="100%" stopColor="rgba(100,200,255,0.15)" />
        </linearGradient>
      </defs>

      {progress > 0.1 && (
        <path
          d={`M ${centerX - 300} 200 L ${centerX - 50 * (1 - progress) - 20} ${600 + progress * 200} L ${centerX + 50 * (1 - progress) + 20} ${600 + progress * 200} L ${centerX + 300} 200 Z`}
          fill="url(#funnelGrad)"
          opacity={Math.min(progress * 2, 0.8)}
        />
      )}

      {/* Converged state indicator */}
      {progress > 0.7 && (
        <circle
          cx={centerX}
          cy={centerY + 100}
          r={30 + (1 - progress) * 50}
          fill="none"
          stroke="#4ECDC4"
          strokeWidth={2}
          opacity={progress}
          strokeDasharray="8 4"
        >
          <animateTransform
            attributeName="transform"
            type="rotate"
            from={`0 ${centerX} ${centerY + 100}`}
            to={`360 ${centerX} ${centerY + 100}`}
            dur="4s"
            repeatCount="indefinite"
          />
        </circle>
      )}

      {/* Brain dots */}
      {states.map((state, i) => {
        const wobble = Math.sin(frame * 0.05 + i * 1.5) * (1 - progress) * 30;
        const x = centerX + state.baseX * (1 - progress * 0.9) + wobble;
        const y = 400 + state.baseY * (1 - progress * 0.7) + progress * 300;
        const activation = 0.3 + Math.sin(frame * 0.08 + i) * 0.3 * (1 - progress) + progress * 0.5;

        return (
          <g key={state.label}>
            <BrainSvg
              activation={activation}
              color={progress > 0.8 ? "#4ECDC4" : state.color}
              x={x}
              y={y}
              size={60 - progress * 10}
            />
            <text
              x={x}
              y={y + 50}
              textAnchor="middle"
              fill={progress > 0.8 ? "#4ECDC4" : state.color}
              fontSize={14}
              fontWeight={700}
              fontFamily="-apple-system, system-ui, sans-serif"
              opacity={1 - progress * 0.7}
            >
              {state.label}
            </text>
          </g>
        );
      })}

      {/* Label */}
      {progress > 0.8 && (
        <text
          x={centerX}
          y={centerY + 180}
          textAnchor="middle"
          fill="#4ECDC4"
          fontSize={20}
          fontWeight={700}
          fontFamily="-apple-system, system-ui, sans-serif"
          opacity={(progress - 0.8) * 5}
        >
          CONVERGED STATE
        </text>
      )}
    </svg>
  );
};

// ── Variance Bar ─────────────────────────────────────────────────────────

const VarianceBar: React.FC<{
  value: number; // 0-100
  label: string;
  color: string;
}> = ({ value, label, color }) => {
  const { fps } = useVideoConfig();
  const frame = useCurrentFrame();

  const animatedValue = spring({
    frame,
    fps,
    config: { damping: 20 },
    from: 100,
    to: value,
  });

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 12 }}>
      <div style={{ width: 140, fontSize: 16, fontWeight: 600, color: "#aaa", textAlign: "right" }}>
        {label}
      </div>
      <div style={{ flex: 1, height: 24, background: "rgba(255,255,255,0.05)", borderRadius: 12 }}>
        <div
          style={{
            width: `${animatedValue}%`,
            height: "100%",
            background: color,
            borderRadius: 12,
            transition: "width 0.3s",
          }}
        />
      </div>
      <div style={{ width: 60, fontSize: 18, fontWeight: 800, color }}>
        {Math.round(animatedValue)}%
      </div>
    </div>
  );
};

// ── Bold Caption ─────────────────────────────────────────────────────────

const Caption: React.FC<{
  text: string;
  durationFrames: number;
  highlights?: string[];
}> = ({ text, durationFrames, highlights = [] }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const words = text.split(" ");
  const wordsPerBatch = 4;
  const batches: string[][] = [];
  for (let i = 0; i < words.length; i += wordsPerBatch) {
    batches.push(words.slice(i, i + wordsPerBatch));
  }

  const framesPerBatch = durationFrames / batches.length;
  const batchIdx = Math.min(Math.floor(frame / framesPerBatch), batches.length - 1);
  const batch = batches[batchIdx];
  const batchFrame = frame - batchIdx * framesPerBatch;

  const opacity = spring({ frame: batchFrame, fps, config: { damping: 15 } });
  const scale = spring({ frame: batchFrame, fps, config: { damping: 12, stiffness: 200 }, from: 0.85, to: 1 });

  return (
    <div
      style={{
        position: "absolute",
        bottom: 160,
        left: 40,
        right: 40,
        display: "flex",
        justifyContent: "center",
        flexWrap: "wrap",
        gap: 10,
        opacity,
        transform: `scale(${scale})`,
      }}
    >
      {batch.map((word, i) => {
        const isHl = highlights.some((h) => word.toLowerCase().includes(h.toLowerCase()));
        return (
          <span
            key={`${batchIdx}-${i}`}
            style={{
              fontSize: 48,
              fontWeight: 900,
              color: isHl ? "#FFD93D" : "white",
              textShadow: "0 0 30px rgba(0,0,0,0.9), 3px 3px 0 rgba(0,0,0,0.7)",
              fontFamily: "-apple-system, system-ui, sans-serif",
              lineHeight: 1.3,
            }}
          >
            {word}
          </span>
        );
      })}
    </div>
  );
};

// ── Number Counter ───────────────────────────────────────────────────────

const BigNumber: React.FC<{
  from: number;
  to: number;
  suffix: string;
  label: string;
  color: string;
}> = ({ from, to, suffix, label, color }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const value = spring({ frame, fps, config: { damping: 20 }, from, to });

  return (
    <div style={{ textAlign: "center", marginTop: 40 }}>
      <div style={{ fontSize: 120, fontWeight: 900, color, fontFamily: "-apple-system, system-ui, sans-serif" }}>
        {Math.round(value)}{suffix}
      </div>
      <div style={{ fontSize: 22, color: "#888", fontWeight: 600, marginTop: -10 }}>
        {label}
      </div>
    </div>
  );
};

// ── Main Composition ─────────────────────────────────────────────────────

export const BrainEntrainment: React.FC = () => {
  const frame = useCurrentFrame();

  const bgHue = interpolate(frame, [0, FPS * 62], [240, 200]);

  const highlights = ["eighty-seven", "breathing", "variance", "wonder", "stars", "four-dimensional", "breath", "somatic", "converged", "reset", "brain"];

  return (
    <AbsoluteFill>
      {/* Background */}
      <AbsoluteFill style={{
        background: `linear-gradient(170deg, hsl(${bgHue}, 35%, 6%) 0%, hsl(${bgHue + 40}, 25%, 3%) 100%)`,
      }} />

      {/* Grid */}
      <AbsoluteFill style={{
        backgroundImage: "linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px)",
        backgroundSize: "50px 50px",
      }} />

      {/* Music */}
      <Audio src={staticFile("audio/bg_music.mp3")} volume={0.12} />

      {/* Title badge */}
      <div style={{
        position: "absolute", top: 50, left: 0, right: 0, textAlign: "center",
        fontSize: 18, fontWeight: 700, color: "#555", letterSpacing: 3,
        fontFamily: "-apple-system, system-ui, sans-serif",
      }}>
        BRAIN ENTRAINMENT
      </div>

      {/* Segment 1: Hook — big number reveal */}
      <Sequence from={0} durationInFrames={5 * FPS}>
        <Audio src={staticFile("audio/hook.mp3")} volume={0.85} />
        <BigNumber from={0} to={87} suffix="%" label="brain state variance reduction" color="#4ECDC4" />
        <Caption text={segments[0].text} durationFrames={5 * FPS} highlights={highlights} />
      </Sequence>

      {/* Segment 2: Setup — show 5 scattered brain states */}
      <Sequence from={5 * FPS} durationInFrames={7 * FPS}>
        <Audio src={staticFile("audio/setup.mp3")} volume={0.85} />
        <EntrainmentFunnel progress={0} />
        <Caption text={segments[1].text} durationFrames={7 * FPS} highlights={highlights} />
      </Sequence>

      {/* Segment 3: Problem — variance is high */}
      <Sequence from={12 * FPS} durationInFrames={6 * FPS}>
        <Audio src={staticFile("audio/problem.mp3")} volume={0.85} />
        <EntrainmentFunnel progress={0.05} />
        <BigNumber from={0} to={91} suffix="" label="units of brain state variance" color="#FF6B6B" />
        <Caption text={segments[2].text} durationFrames={6 * FPS} highlights={highlights} />
      </Sequence>

      {/* Segment 4: Solution — funnel animation converging */}
      <Sequence from={18 * FPS} durationInFrames={10 * FPS}>
        <Audio src={staticFile("audio/solution.mp3")} volume={0.85} />
        {(() => {
          const localFrame = frame - 18 * FPS;
          const progress = interpolate(localFrame, [0, 10 * FPS], [0, 1], { extrapolateRight: "clamp", extrapolateLeft: "clamp" });
          return <EntrainmentFunnel progress={Math.max(0, progress)} />;
        })()}
        <Caption text={segments[3].text} durationFrames={10 * FPS} highlights={highlights} />
      </Sequence>

      {/* Segment 5: Why — bar chart of techniques */}
      <Sequence from={28 * FPS} durationInFrames={8 * FPS}>
        <Audio src={staticFile("audio/why.mp3")} volume={0.85} />
        <div style={{ position: "absolute", top: 200, left: 60, right: 60 }}>
          <div style={{ fontSize: 24, fontWeight: 800, color: "white", marginBottom: 30, textAlign: "center" }}>
            Variance Reduction by Technique
          </div>
          <VarianceBar value={87} label="Breath Focus" color="#4ECDC4" />
          <VarianceBar value={86} label="Body Scan" color="#45B7AA" />
          <VarianceBar value={80} label="Counting" color="#3A9B8F" />
          <VarianceBar value={67} label="Sensory" color="#2D7D73" />
        </div>
        <Caption text={segments[4].text} durationFrames={8 * FPS} highlights={highlights} />
      </Sequence>

      {/* Segment 6: Wonder — stars / cosmic */}
      <Sequence from={36 * FPS} durationInFrames={10 * FPS}>
        <Audio src={staticFile("audio/wonder.mp3")} volume={0.85} />
        {/* Starfield */}
        <AbsoluteFill>
          {Array.from({ length: 80 }, (_, i) => {
            const x = ((i * 137.5) % 1080);
            const y = ((i * 89.3) % 1200) + 200;
            const size = 1 + (i % 3);
            const twinkle = Math.sin(frame * 0.1 + i) * 0.5 + 0.5;
            return (
              <div
                key={i}
                style={{
                  position: "absolute",
                  left: x,
                  top: y,
                  width: size,
                  height: size,
                  borderRadius: "50%",
                  background: "white",
                  opacity: twinkle * 0.8,
                }}
              />
            );
          })}
        </AbsoluteFill>
        <Img
          src={staticFile("brains/affective_steering.png")}
          style={{
            position: "absolute",
            top: 250,
            left: 60,
            right: 60,
            width: 960,
            borderRadius: 20,
            opacity: 0.85,
          }}
        />
        <Caption text={segments[5].text} durationFrames={10 * FPS} highlights={highlights} />
      </Sequence>

      {/* Segment 7: Dynamics — 4D manifold */}
      <Sequence from={46 * FPS} durationInFrames={8 * FPS}>
        <Audio src={staticFile("audio/dynamics.mp3")} volume={0.85} />
        <BigNumber from={20000} to={4} suffix="D" label="effective dimensionality of brain dynamics" color="#FFD93D" />
        <Img
          src={staticFile("brains/state_velocity.png")}
          style={{
            position: "absolute",
            top: 350,
            left: 40,
            right: 40,
            width: 1000,
            borderRadius: 16,
            opacity: 0.9,
          }}
        />
        <Caption text={segments[6].text} durationFrames={8 * FPS} highlights={highlights} />
      </Sequence>

      {/* Segment 8: CTA — converged + stars */}
      <Sequence from={54 * FPS} durationInFrames={8 * FPS}>
        <Audio src={staticFile("audio/cta.mp3")} volume={0.85} />
        <EntrainmentFunnel progress={1} />
        <Caption text={segments[7].text} durationFrames={8 * FPS} highlights={highlights} />
      </Sequence>
    </AbsoluteFill>
  );
};
