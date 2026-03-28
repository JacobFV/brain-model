import React from "react";
import { useCurrentFrame } from "remotion";

/**
 * Animated SVG talking face placeholder for A-roll.
 * Mouth opens/closes in a speech pattern synced to frame count.
 */
export const SvgFace: React.FC<{ speaking?: boolean }> = ({
  speaking = true,
}) => {
  const frame = useCurrentFrame();

  // Mouth animation — oscillate between open and closed
  const mouthOpen = speaking
    ? 4 + Math.sin(frame * 0.5) * 3 + Math.sin(frame * 1.3) * 2
    : 1;
  const mouthWidth = 18 + (speaking ? Math.sin(frame * 0.7) * 4 : 0);

  // Subtle head bob
  const headY = Math.sin(frame * 0.08) * 2;
  const headTilt = Math.sin(frame * 0.06) * 1.5;

  // Blink every ~90 frames
  const blinkPhase = frame % 90;
  const eyeHeight = blinkPhase < 3 ? 0.5 : 5;

  return (
    <svg
      viewBox="0 0 200 200"
      style={{ width: "100%", height: "100%" }}
    >
      <g transform={`translate(100, ${100 + headY}) rotate(${headTilt})`}>
        {/* Head */}
        <ellipse cx={0} cy={0} rx={70} ry={80} fill="#F5D6B8" />

        {/* Hair */}
        <ellipse cx={0} cy={-45} rx={72} ry={40} fill="#3D2B1F" />
        <rect x={-72} y={-50} width={144} height={10} fill="#3D2B1F" rx={5} />

        {/* Eyebrows */}
        <line x1={-30} y1={-22} x2={-12} y2={-25} stroke="#3D2B1F" strokeWidth={2.5} strokeLinecap="round" />
        <line x1={12} y1={-25} x2={30} y2={-22} stroke="#3D2B1F" strokeWidth={2.5} strokeLinecap="round" />

        {/* Eyes */}
        <ellipse cx={-22} cy={-12} rx={5} ry={eyeHeight} fill="#2C2C2C" />
        <ellipse cx={22} cy={-12} rx={5} ry={eyeHeight} fill="#2C2C2C" />

        {/* Eye whites / glint */}
        {eyeHeight > 2 && (
          <>
            <circle cx={-20} cy={-14} r={1.5} fill="white" />
            <circle cx={24} cy={-14} r={1.5} fill="white" />
          </>
        )}

        {/* Nose */}
        <path d="M 0 -2 Q 5 8 0 12 Q -5 8 0 -2" fill="none" stroke="#D4A574" strokeWidth={1.5} />

        {/* Mouth */}
        <ellipse
          cx={0}
          cy={26}
          rx={mouthWidth / 2}
          ry={mouthOpen}
          fill={mouthOpen > 3 ? "#8B1A1A" : "#C4836E"}
          stroke="#B5735E"
          strokeWidth={1}
        />

        {/* Teeth hint when mouth open */}
        {mouthOpen > 4 && (
          <rect x={-8} y={23} width={16} height={3} fill="white" rx={1} />
        )}

        {/* Ears */}
        <ellipse cx={-68} cy={-5} rx={8} ry={12} fill="#EDCAA8" />
        <ellipse cx={68} cy={-5} rx={8} ry={12} fill="#EDCAA8" />
      </g>
    </svg>
  );
};
