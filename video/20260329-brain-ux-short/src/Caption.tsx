import React from "react";
import {
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";

/**
 * Bold animated caption — words pop in with spring animation.
 * Style: large white text, black outline, centered.
 */
export const Caption: React.FC<{
  text: string;
  startFrame: number;
  durationFrames: number;
  highlight?: string[];
}> = ({ text, startFrame, durationFrames, highlight = [] }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const relFrame = frame - startFrame;
  if (relFrame < 0 || relFrame > durationFrames) return null;

  const words = text.split(" ");
  const wordsPerBatch = 4;
  const batches: string[][] = [];
  for (let i = 0; i < words.length; i += wordsPerBatch) {
    batches.push(words.slice(i, i + wordsPerBatch));
  }

  const framesPerBatch = durationFrames / batches.length;
  const currentBatchIdx = Math.min(
    Math.floor(relFrame / framesPerBatch),
    batches.length - 1
  );
  const currentBatch = batches[currentBatchIdx];
  const batchRelFrame = relFrame - currentBatchIdx * framesPerBatch;

  // Fade in spring
  const opacity = spring({
    frame: batchRelFrame,
    fps,
    config: { damping: 15 },
  });

  const scale = spring({
    frame: batchRelFrame,
    fps,
    config: { damping: 12, stiffness: 200 },
    from: 0.8,
    to: 1,
  });

  return (
    <div
      style={{
        position: "absolute",
        bottom: 180,
        left: 40,
        right: 40,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexWrap: "wrap",
        gap: 8,
        opacity,
        transform: `scale(${scale})`,
      }}
    >
      {currentBatch.map((word, i) => {
        const isHighlight = highlight.some((h) =>
          word.toLowerCase().includes(h.toLowerCase())
        );
        return (
          <span
            key={`${currentBatchIdx}-${i}`}
            style={{
              fontSize: 52,
              fontWeight: 900,
              color: isHighlight ? "#FF4444" : "white",
              textShadow: "0 0 20px rgba(0,0,0,0.9), 2px 2px 0 rgba(0,0,0,0.8)",
              fontFamily: "-apple-system, system-ui, sans-serif",
              lineHeight: 1.2,
              textAlign: "center",
            }}
          >
            {word}
          </span>
        );
      })}
    </div>
  );
};
