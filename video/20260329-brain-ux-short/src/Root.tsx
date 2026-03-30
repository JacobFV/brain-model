import { Composition } from "remotion";
import { BrainUxShort } from "./BrainUxShort";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="BrainUxShort"
        component={BrainUxShort}
        durationInFrames={30 * 58}
        fps={30}
        width={1080}
        height={1920}
      />
    </>
  );
};
