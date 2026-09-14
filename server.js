// 営業ボイス議事録 - サーバー
//
// このサーバーがやっていることは1つだけです:
//   1. iPhoneのブラウザから送られてきた録音データ(base64)を受け取る
//   2. Gemini APIに「文字起こし + 要約」を頼む
//   3. 結果をブラウザに返す
//
// 商談の記録そのもの(文字起こし・要約・カレンダー)は保存せず、
// ブラウザ(iPhone)側のlocalStorageに保存する設計にしています。
// そのため、このサーバーは再起動してもデータが消える心配がありません
// (そもそも何も保存していないので)。

const express = require("express");
const path = require("path");
const dotenv = require("dotenv");
const { GoogleGenAI, Type } = require("@google/genai");

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// 録音データ(base64)を受け取れるようにアップロード上限を大きめに設定
app.use(express.json({ limit: "30mb" }));
app.use(express.urlencoded({ limit: "30mb", extended: true }));

// 静的ファイル(フロントエンド)を配信
app.use(express.static(path.join(__dirname, "public")));

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const ai = GEMINI_API_KEY ? new GoogleGenAI({ apiKey: GEMINI_API_KEY }) : null;

// 切削工具(エンドミル)営業が扱う専門用語・日進工具(NS TOOL)の製品名の辞書。
// 音声認識でカタカナ・型番が崩れやすいので、Geminiに前もって渡しておく。
const DOMAIN_GLOSSARY = `
【業界知識コンテキスト】
この商談は切削工具、特にエンドミル(切削工具)メーカー「日進工具(NS TOOL)」に関するものです。
音声を文字起こしする際、以下のような専門用語・製品名が使われている可能性が高いです。
発音が不明瞭だったり、誤変換されそうな場合は、これらの正しい表記に補正してください。

- 日進工具(NS TOOL)の主な製品・技術: 無限コーティング(MUGEN-COATING)、無限コーティングプレミアム、
  マイクロエンドミル、超硬エンドミル(スクエア・ボール・ラジアス・テーパー)、
  MSE230、MSBH230、MTB230 などの型番シリーズ
- 加工・技術用語: エンドミル、マシニングセンタ、被削材、金型用鋼、高硬度鋼、調質鋼、ステンレス、
  チタン、アルミ、NAK80、チッピング(刃こぼれ)、摩耗、クーラント(切削油)、
  送り速度(Feed)、回転数(Spindle)、切り込み量(ap/ae)、面粗度、首下長、有効長、超硬合金、シャンク
`.trim();

const RESPONSE_SCHEMA = {
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
};

function normalizeMimeType(mimeType) {
  const m = (mimeType || "").toLowerCase();
  if (m.includes("mp4") || m.includes("m4a") || m.includes("aac") || m.includes("caf")) return "audio/mp4";
  if (m.includes("webm")) return "audio/webm";
  if (m.includes("wav")) return "audio/wav";
  if (m.includes("ogg") || m.includes("opus")) return "audio/ogg";
  if (m.includes("mp3") || m.includes("mpeg")) return "audio/mpeg";
  if (m.includes("flac")) return "audio/flac";
  return "audio/mp4";
}

function safeParseJson(text) {
  let raw = (text || "").trim();
  if (raw.startsWith("```")) {
    raw = raw.replace(/^```(json)?\s*/i, "").replace(/\s*```$/, "");
  }
  return JSON.parse(raw.trim());
}

app.post("/api/transcribe", async (req, res) => {
  try {
    if (!ai) {
      return res.status(500).json({
        error: "サーバーにGEMINI_API_KEYが設定されていません。デプロイ先の環境変数を確認してください。",
      });
    }

    const { audio, mimeType } = req.body || {};
    if (!audio) {
      return res.status(400).json({ error: "音声データが送られてきませんでした。" });
    }

    const cleanMimeType = normalizeMimeType(mimeType);

    const audioPart = {
      inlineData: {
        data: audio,
        mimeType: cleanMimeType,
      },
    };

    const promptPart = {
      text: `
添付された音声ファイルを聴き、できるだけ正確に日本語で書き起こし、内容を分析して指定のJSON形式で返してください。

${DOMAIN_GLOSSARY}

抽出項目:
- userName: 話している営業担当者(自分)の名前。会話内で名乗っていなければ「確認中」としてください。
- clientName: 商談相手(会社名・役職・氏名など分かる範囲)。不明なら「確認中」としてください。
- summaryPoints: 商談内容の要点を、丁寧な日本語の箇条書きで3〜6項目に整理してください。
- transcription: 音声のほぼ正確な全文の書き起こし。専門用語や固有名詞は、上記の表記を優先して補正してください。

JSON以外の文章は出力しないでください。
`.trim(),
    };

    const response = await ai.models.generateContent({
      model: "gemini-3.8-flash",
      contents: [audioPart, promptPart],
      config: {
        responseMimeType: "application/json",
        responseSchema: RESPONSE_SCHEMA,
      },
    });

    if (!response.text) {
      throw new Error("Geminiから解析結果を受け取れませんでした。");
    }

    const result = safeParseJson(response.text);
    res.json(result);
  } catch (error) {
    console.error("Transcription error:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "サーバーで不明なエラーが発生しました。",
    });
  }
});

// シンプルな死活監視用エンドポイント(Renderなどのヘルスチェックに使える)
app.get("/healthz", (req, res) => res.send("ok"));

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  if (!GEMINI_API_KEY) {
    console.warn("警告: GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。");
  }
});
