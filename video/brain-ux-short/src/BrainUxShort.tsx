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
} from "remotion";
import { SvgFace } from "./SvgFace";
import { Caption } from "./Caption";
import scriptData from "../public/script.json";

const FPS = 30;
const segments = scriptData.segments;

/** Gradient background that slowly shifts */
const Background: React.FC = () => {
  const frame = useCurrentFrame();
  const hue = interpolate(frame, [0, FPS * 58], [220, 280]);
  return (
    <AbsoluteFill
      style={{
        background: `linear-gradient(170deg, hsl(${hue}, 30%, 8%) 0%, hsl(${hue + 30}, 25%, 4%) 100%)`,
      }}
    />
  );
};

/** Subtle grid overlay for tech feel */
const GridOverlay: React.FC = () => (
  <AbsoluteFill
    style={{
      backgroundImage:
        "linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)",
      backgroundSize: "40px 40px",
    }}
  />
);

/** B-roll: brain image with ken-burns effect */
const BrainVisual: React.FC<{
  src: string;
  durationFrames: number;
}> = ({ src, durationFrames }) => {
  const frame = useCurrentFrame();
  const scale = interpolate(frame, [0, durationFrames], [1, 1.08], {
    extrapolateRight: "clamp",
  });
  const y = interpolate(frame, [0, durationFrames], [0, -20], {
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        position: "absolute",
        top: 120,
        left: 40,
        right: 40,
        height: 900,
        borderRadius: 24,
        overflow: "hidden",
        border: "2px solid rgba(255,255,255,0.1)",
      }}
    >
      <Img
        src={staticFile(src)}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          transform: `scale(${scale}) translateY(${y}px)`,
        }}
      />
    </div>
  );
};

/** A-roll: face-to-camera with SVG avatar */
const FaceToCamera: React.FC<{ durationFrames: number }> = ({
  durationFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const fadeIn = spring({ frame, fps, config: { damping: 15 } });
  const fadeOut = frame > durationFrames - 8
    ? interpolate(frame, [durationFrames - 8, durationFrames], [1, 0], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      })
    : 1;

  return (
    <AbsoluteFill
      style={{
        opacity: fadeIn * fadeOut,
      }}
    >
      {/* Camera-style frame */}
      <div
        style={{
          position: "absolute",
          top: 200,
          left: 140,
          right: 140,
          height: 800,
          background: "linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)",
          borderRadius: 30,
          overflow: "hidden",
          border: "3px solid rgba(255,255,255,0.1)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div style={{ width: 400, height: 400 }}>
          <SvgFace speaking={true} />
        </div>

        {/* Camera REC indicator */}
        <div
          style={{
            position: "absolute",
            top: 20,
            right: 20,
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: "#ff3333",
              opacity: Math.sin(frame * 0.1) > 0 ? 1 : 0.3,
            }}
          />
          <span
            style={{
              color: "#ff3333",
              fontSize: 14,
              fontWeight: 700,
              fontFamily: "monospace",
            }}
          >
            REC
          </span>
        </div>
      </div>
    </AbsoluteFill>
  );
};

/** Title card */
const TitleCard: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const titleSpring = spring({ frame, fps, config: { damping: 12 } });

  return (
    <div
      style={{
        position: "absolute",
        top: 60,
        left: 40,
        right: 40,
        textAlign: "center",
        transform: `translateY(${(1 - titleSpring) * -30}px)`,
        opacity: titleSpring,
      }}
    >
      <div
        style={{
          fontSize: 22,
          fontWeight: 700,
          color: "#666",
          letterSpacing: 4,
          marginBottom: 8,
          fontFamily: "-apple-system, system-ui, sans-serif",
        }}
      >
        META TRIBE v2
      </div>
      <div
        style={{
          fontSize: 18,
          color: "#444",
          fontFamily: "-apple-system, system-ui, sans-serif",
        }}
      >
        fMRI BRAIN PREDICTION
      </div>
    </div>
  );
};

export const BrainUxShort: React.FC = () => {
  return (
    <AbsoluteFill>
      <Background />
      <GridOverlay />

      {/* Background music — low volume, full duration */}
      <Audio src={staticFile("audio/bg_music.mp3")} volume={0.15} />

      {/* Narration segments */}
      {segments.map((seg) => {
        const startFrame = Math.round(seg.startSec * FPS);
        const durationFrames = Math.round((seg.endSec - seg.startSec) * FPS);

        return (
          <Sequence
            key={seg.id}
            from={startFrame}
            durationInFrames={durationFrames}
          >
            {/* Narration audio for this segment */}
            <Audio src={staticFile(`audio/${seg.id}.mp3`)} volume={0.9} />

            {/* Title badge (always visible) */}
            <TitleCard />

            {/* Visual layer */}
            {seg.type === "broll" && seg.visual && (
              <BrainVisual src={seg.visual} durationFrames={durationFrames} />
            )}
            {seg.type === "aroll" && (
              <FaceToCamera durationFrames={durationFrames} />
            )}

            {/* Captions */}
            <Caption
              text={seg.text}
              startFrame={0}
              durationFrames={durationFrames}
              highlight={[
                "reward",
                "dark",
                "hijack",
                "doubles",
                "brain",
                "flow",
                "trust",
                "spatial",
                "cognitive",
              ]}
            />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
