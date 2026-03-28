import { Composition } from "remotion";
import { BrainEntrainment } from "./BrainEntrainment";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="BrainEntrainment"
      component={BrainEntrainment}
      durationInFrames={30 * 62}
      fps={30}
      width={1080}
      height={1920}
    />
  );
};
