import OpenAI from "openai";
import fs from "fs";
import path from "path";

const script = JSON.parse(
  fs.readFileSync(path.join(__dirname, "public", "script.json"), "utf-8")
);

const client = new OpenAI();

async function main() {
  const audioDir = path.join(__dirname, "public", "audio");
  fs.mkdirSync(audioDir, { recursive: true });

  for (const seg of script.segments) {
    const outPath = path.join(audioDir, `${seg.id}.mp3`);
    if (fs.existsSync(outPath)) {
      console.log(`Skip (exists): ${seg.id}`);
      continue;
    }

    console.log(`Generating: ${seg.id} — "${seg.text.slice(0, 60)}..."`);
    const response = await client.audio.speech.create({
      model: "tts-1-hd",
      voice: "onyx",
      input: seg.text,
      speed: 1.1,
    });

    const buffer = Buffer.from(await response.arrayBuffer());
    fs.writeFileSync(outPath, buffer);
    console.log(`  Saved: ${outPath} (${(buffer.length / 1024).toFixed(0)} KB)`);
  }

  // Generate full narration as one file too
  const fullText = script.segments.map((s: any) => s.text).join(" ");
  const fullPath = path.join(audioDir, "full_narration.mp3");
  if (!fs.existsSync(fullPath)) {
    console.log("Generating full narration...");
    const response = await client.audio.speech.create({
      model: "tts-1-hd",
      voice: "onyx",
      input: fullText,
      speed: 1.1,
    });
    const buffer = Buffer.from(await response.arrayBuffer());
    fs.writeFileSync(fullPath, buffer);
    console.log(`  Saved: ${fullPath} (${(buffer.length / 1024).toFixed(0)} KB)`);
  }

  console.log("Done!");
}

main().catch(console.error);
