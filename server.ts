import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

// Initialize Express
const app = express();
const PORT = 3000;

// Allow large base64 audio uploads
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));

// Initialize Gemini SDK with telemetry header
const geminiApiKey = process.env.GEMINI_API_KEY;
const ai = new GoogleGenAI({
  apiKey: geminiApiKey || "",
  httpOptions: {
    headers: {
      "User-Agent": "aistudio-build",
    },
  },
});

// Transcription and summarization API
app.post("/api/transcribe-summarize", async (req, res) => {
  try {
    const { audio, mimeType, method } = req.body;

    if (!audio) {
      return res.status(400).json({ error: "音声データがありません。" });
    }

    const audioBuffer = Buffer.from(audio, "base64");
    const openaiKey = process.env.OPENAI_API_KEY;

    // Normalize and clean mimeType to support a wide list of devices & keep Gemini/Whisper requests happy
    const originalMimeType = mimeType || "audio/mp4";
    let cleanMimeType = originalMimeType.toLowerCase().trim();
    
    if (cleanMimeType.includes("m4a") || cleanMimeType.includes("x-m4a") || cleanMimeType.includes("caf") || cleanMimeType.includes("x-caf") || cleanMimeType.includes("mp4") || cleanMimeType.includes("aac")) {
      cleanMimeType = "audio/mp4";
    } else if (cleanMimeType.includes("mp3") || cleanMimeType.includes("mpeg")) {
      cleanMimeType = "audio/mpeg";
    } else if (cleanMimeType.includes("wav") || cleanMimeType.includes("x-wav")) {
      cleanMimeType = "audio/wav";
    } else if (cleanMimeType.includes("webm")) {
      cleanMimeType = "audio/webm";
    } else if (cleanMimeType.includes("ogg") || cleanMimeType.includes("opus")) {
      cleanMimeType = "audio/ogg";
    } else if (cleanMimeType.includes("flac")) {
      cleanMimeType = "audio/flac";
    } else {
      cleanMimeType = "audio/mp4"; // Default fallback
    }

    console.log(`Audio File Upload - Original MIME: "${originalMimeType}" -> Normalized for Gemini: "${cleanMimeType}"`);

    // Safe helper to parse JSON with potential markdown backticks from LLM output
    const cleanAndParseJson = (text: string) => {
      let raw = text.trim();
      if (raw.startsWith("```")) {
        raw = raw.replace(/^```(json)?\s*/i, "").replace(/\s*```$/, "");
      }
      return JSON.parse(raw.trim());
    };

    let transcriptionText = "";

    // If Whisper is requested and we have the OpenAI key, use Whisper for transcription
    if (method === "whisper" && openaiKey) {
      console.log("Using OpenAI Whisper API for transcription...");
      
      const formData = new FormData();
      // Safe file name mapping based on cleaned mimetypes
      const ext = cleanMimeType.includes("mp4") ? "mp4" : cleanMimeType.includes("webm") ? "webm" : "wav";
      const file = new File([audioBuffer], `audio.${ext}`, { type: cleanMimeType });
      formData.append("file", file);
      formData.append("model", "whisper-1");
      formData.append("language", "ja");

      const whisperResponse = await fetch("https://api.openai.com/v1/audio/transcriptions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${openaiKey}`,
        },
        body: formData,
      });

      if (!whisperResponse.ok) {
        const errText = await whisperResponse.text();
        throw new Error(`Whisper API error: ${errText}`);
      }

      const whisperResult = await whisperResponse.json();
      transcriptionText = whisperResult.text;
      
      console.log("Whisper transcription completed, summarizing with Gemini...");
    }

    // Now use Gemini to either:
    // A) Transcribe + Summarize directly (if no Whisper key, or "gemini" selected)
    // B) Summarize the text transcribed by Whisper
    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({ error: "Gemini APIキーが設定されていません。" });
    }

    let result;

    if (transcriptionText) {
      // B) Summarize and structure the pre-transcribed text utilizing Gemini
      const prompt = `
以下のテキストは商談の文字起こしです。この内容を分析し、指定されたJSON構造で返してください。

【業界知識コンテキスト（最重要）】
この商談は切削工具（特に「日進工具（NS TOOL）」の「無限コーティング（MUGEN-COATING）」「マイクロエンドミル」「超硬エンドミル」や、高硬度鋼加工、被削材・切削条件など）に関するものです。
文字起こし内の専門用語や、誤変換・曖昧な単語表記を脳内で「切削加工・製造業の適切な用語」に読み替えて正確に解釈し、要約や補正に反映してください：
- 主な製品・技術: 無限コーティング（MUGEN-COATING）、無限コーティングプレミアム、無限プレミアム、マイクロエンドミル、超硬エンドミル（スクエア・ボール・ラジアス・テーパー）、MSE230、MSBH230 などの日進工具ブランド
- 加工・技術用語: エンドミル、マシニングセンタ、被削材（金型用鋼、高硬度鋼、調質鋼、ステンレス、アルミ）、チッピング（刃こぼれ）、摩耗、クーラント（切削油）、送り速度（Feed）、回転数（Spindle）、切り込み量（ap/ae）、面粗度、超硬、金型加工

文字起こしテキスト:
"""
${transcriptionText}
"""

要約条件:
- 担当者（ユーザー）の名前 (userName) を抽出（不明なら「確認中」）
- 商談相手の会社名・役職・名前 (clientName) を抽出（不明なら「確認中」）
- 商談内容のキーポイント・議事録 (summaryPoints) を、箇条書き（3〜5項目）で整理
- 文字起こしテキスト全体 (transcription) は、渡された文字起こしをベースに、切削工具や製造業関連 of 誤変換があれば正しく自然な専門用語に補正して設定

日本語で出力してください。
`;

      const response = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              transcription: { type: Type.STRING },
              userName: { type: Type.STRING },
              clientName: { type: Type.STRING },
              summaryPoints: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
            },
            required: ["transcription", "userName", "clientName", "summaryPoints"],
          },
        },
      });

      if (!response.text) {
        throw new Error("Gemini から要約結果を受け取れませんでした。");
      }

      result = cleanAndParseJson(response.text);
    } else {
      // A) Transcribe and Summarize directly using Gemini's native Audio capability!
      console.log("Using Gemini native Audio feature for transcription and summary...");
      
      const audioPart = {
        inlineData: {
          data: audio,
          mimeType: cleanMimeType,
        },
      };

      const promptPart = {
        text: `
添付された音声ファイルを聴き、音声を高精度に書き起こして、さらに内容を分析して指定されたJSON構造で返してください。

【業界特有の専門用語補正辞書（最重要）】
音声をテキストに書き起こす際、以下のような製造業・切削加工（特に「日進工具（NS TOOL）」関連）の専門用語が使われている可能性が�    res.json(result);
  } catch (error) {
    console.error("Transcription error:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "サーバーでエラーが発生しました。",
    });
  }
});hema: {
            type: Type.OBJECT,
            properties: {
              transcription: { type: Type.STRING },
              userName: { type: Type.STRING },
              clientName: { type: Type.STRING },
              summaryPoints: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
            },
            required: ["transcription", "userName", "clientName", "summaryPoints"],
          },
        },
      });

      if (!response.text) {
        throw new Error("Gemini から解析結果を受け取れませんでした。");
      }

      result = cleanAndParseJson(response.text);
    }

    res.json(result);
  } catch (error) {
    console.error("Transcription error:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "サーバーでエラーが発生しました。",
    });
  }
});�、超硬合金、シャンク、チッピング、マシニングセンタ、クーラント（切削油）、切り込み量（ap/ae）、面粗度、摩耗、首下長、有効長
- 日進工具（NS TOOL）関連: 無限コーティング（MUGEN-COATING）、無限コーティングプレミアム（MUGEN-COATING PREMIUM）、マイクロエンドミル、無限プレミアム、無限ミニチュア、MSE230、MSBH230、MTB230 などの製品型番
- 被削材（削る対象）: 高硬度鋼、調質鋼、ステンレス、チタン、銅、アルミ、金型用鋼（NAK80など）
- 加工条件: 回転数（Spindle）、送り速度（Feed）

抽出項目:
- 担当者（ユーザー）の名前 (userName): 会話内での営業担当者の名前。不明なら「確認中」
- 商談相手の会社名・代表者・役職など (clientName): 商談先の情報。不明なら「確認中」
- 商談内容の要点 (summaryPoints): 商談の決定事項、次回アクション、課題など。上記の専門用語を理解した上で、箇条書きで3〜5個程度に明瞭に整理
- 全体の文字起こし (transcription): 音声のほぼ正確な全文書き起こし。業界専門用語や固有名詞は、必ず上記の正しい表記を活用して正確に変換してください。

すべて日本語で分かりやすく、かつ丁寧なビジネス表現で作成してください。
`,
      };

      const response = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: [audioPart, promptPart],
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              transcription: { type: Type.STRING },
              userName: { type: Type.STRING },
              clientName: { type: Type.STRING },
              summaryPoints: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
            },
            required: ["transcription", "userName", "clientName", "summaryPoints"],
          },
        },
      });

      if (!response.text) {
        throw new Error("Gemini から解析結果を受け取れませんでした。");
      }

      result = JSON.parse(response.text.trim());
    }

    res.json(result);
  } catch (error) {
    console.error("Transcription error:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "サーバーでエラーが発生しました。",
    });
  }
});

// Vite middleware and Static file serving
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
