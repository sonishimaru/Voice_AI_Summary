# Voice AI Summary

一日中の会話を Mac で常時録音し、**ローカルで日本語文字起こし**、**Claude で要約**して、**毎日 Slack / メールに配信**する個人用システム。

```
[macOS メニューバー常駐アプリ]  マイク + システム音声（Zoom/Meet 等の相手側）を別トラックで録音
        │  15 分ごとに .m4a + .json を inbox/ へ
        ▼
[vas worker]  Silero VAD で無音除去 → kotoba-whisper（faster-whisper）で文字起こし → SQLite(FTS5)
        │
        ▼
[vas digest --deliver]（毎日 22:00, launchd）  エピソード分割 → Claude で要約 → Slack / メール
```

設計上のポイント:

- **音声は外に出ない**。文字起こしは Mac 内で完結し、Claude API に送るのはテキストのみ。
- **マイクとシステム音声を別ファイルに録る**ので、通話では「自分＝`me` / 相手＝`other`」が話者分離モデルなしで分かる。
- システム音声は Core Audio プロセスタップ（macOS 14.4+）で取得。画面収録の権限は不要で、毎月の再許可ダイアログも出ない。
- 検索は SQLite FTS5 の `trigram` トークナイザ（日本語の部分一致が効く）。

## 構成

| パス | 役割 |
|---|---|
| `clients/macos/` | Swift 製メニューバー録音アプリ（xcodegen で Xcode プロジェクト生成） |
| `src/voice_ai_summary/` | Python パッケージ。CLI は `vas` |
| `launchd/` | 常駐ワーカーと日次配信の LaunchAgent 例 |
| `config.example.toml` | 設定ファイルの雛形 |

主なモジュール: `ingest.py`（取り込み）→ `vad.py` / `asr.py` / `pipeline.py`（文字起こし）→ `episodes.py` / `summarize.py`（要約）→ `deliver/`（配信）。

## セットアップ（Mac）

### 1. Python 側

```bash
brew install uv
git clone <this repo> && cd Voice_AI_Summary
uv venv && uv pip install -e ".[dev]"
mkdir -p ~/.config/voice-ai-summary
cp config.example.toml ~/.config/voice-ai-summary/config.toml   # 必要に応じて編集
.venv/bin/vas status
```

秘密情報は環境変数のみで渡します（`~/.zshenv` などに）:

```bash
export VAS_ANTHROPIC_API_KEY=...          # 要約・校正用（vas 専用。ANTHROPIC_API_KEY でも可）
export VAS_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...   # Slack 配信を使う場合
export VAS_SMTP_PASSWORD=...              # メール配信を使う場合（Gmail はアプリパスワード）
```

初回の `vas process` で ASR モデル（`kotoba-tech/kotoba-whisper-v2.0-faster`、約 1.5GB）をダウンロードします。

### 2. 録音アプリ

`clients/macos/README.md` の手順で Xcode プロジェクトを生成・ビルドして起動します。初回起動時に「マイク」と「システムオーディオ録音」の許可を求められます。

### 3. 常駐化

```bash
.venv/bin/vas install-launchd     # worker 常駐 + 毎日 22:00（config で変更可）に digest --deliver
```

## 日常の使い方

```bash
vas status                      # 取り込み・処理状況
vas search 見積もり               # 全文検索（日本語 OK）
vas search 会議 --day 2026-09-15
vas episodes --day 2026-09-15    # その日のエピソード一覧
vas digest --day 2026-09-15      # 要約を生成して表示（配信しない）
vas digest --deliver             # 今日分を生成して Slack / メールへ
vas ingest ~/Downloads/zoom_recording.m4a   # 手持ちの録音を取り込む（Zoom クラウド録画など）
```

## 動作確認（Mac）

1. 録音アプリを起動し、Zoom か動画を再生しながら数分喋る → `~/Library/Application Support/VoiceAISummary/inbox/` に `mac_mic_*.m4a` と `mac_system_*.m4a` ができる（15 分ごと、または一時停止時）。
2. `vas worker --once` → `vas status` で `recordings` が増え、`vas search <喋った単語>` でヒットする。
3. `vas digest --deliver` で Slack / メールに日本語サマリが届く。

## API 費用の確認

vas が行った Claude API 呼び出しはすべて `<data_dir>/usage.jsonl` に記録され、`vas usage` で用途・モデル別のトークン数と概算費用を確認できます。

```bash
vas usage --days 30
```

**注意**: `ANTHROPIC_API_KEY` をシェル全体に export すると、同じ Mac で動かす Claude Code などもそのキーで（サブスクリプションではなく API 従量課金で）動きます。vas には専用の `VAS_ANTHROPIC_API_KEY` を使い、`ANTHROPIC_API_KEY` は必要なときだけ設定してください。

## 文字起こし精度のチューニング

精度が物足りないときは、上から順に試してください（`~/.config/voice-ai-summary/config.toml`）。設定を変えたら `vas reprocess --day YYYY-MM-DD` で再文字起こしして比較できます。

1. **語彙ヒント** `[asr] hotwords = ["西丸", "安田さん", "社名"]` — 人名・製品名の誤変換に最も効く。
2. **チャンク結合** `[vad] merge_gap_ms`（既定 2000）— 短い間で切らず、文単位で Whisper に渡す。
3. **モデル** `[asr] model = "large-v3-turbo"` — kotoba-whisper より遅いが固有名詞や長文に強いことが多い。CPU では `compute_type = "int8"` 推奨。
4. **MLX バックエンド（Apple Silicon の GPU）** `uv pip install -e ".[mlx]"` のうえ `[asr] backend = "mlx"` — `whisper-large-v3-turbo` を GPU で回せるので、大きいモデルが現実的な速度になる。

音声側では、録音アプリのビットレート（32kbps）を上げるより、Zoom 側の「オリジナルサウンド」を有効にする方が効きます。

## Claude による校正と用語集

ローカルの Whisper 文字起こしは、固有名詞や同音異義語（「かたまです」→本来は人名、「ギョウ太郎」「行太郎」のような表記ゆれ）を取りこぼします。音声は Mac の外に出しませんが、テキストだけを Claude に送って校正する仕組みがあります。

```
文字起こし（Whisper） → vas correct（Claude で校正） → vas digest / summarize（要約）
```

- `vas correct --day YYYY-MM-DD` — その日の未校正の発話を Claude（既定 `claude-haiku-4-5`、`[correct]` で変更可）に送り、明らかな認識ミスだけを修正します。口調・方言・フィラーは変更されません。修正前のテキストは `raw_text` 列に残るので、いつでも元の ASR 出力に戻れます。`--show` で修正前後の差分を表示、`--force` で再校正します。
- `vas digest` / `vas summarize` は要約前に自動でこの校正パスを実行します（`[correct] enabled = false` で無効化可能）。要約は文字起こし内容のハッシュでキャッシュされるため、校正でテキストが変わると要約も自動的に再生成されます。
- 校正・要約・ASR の語彙ヒント（`[asr] hotwords`）は、いずれも個人用の **用語集**（人名・社名・製品名や、ユーザー自身の表記の流儀）を参照します。用語集は `<data_dir>/glossary.json` に保存されます。

用語集は手で追加するか、自分の Slack 発言から自動抽出できます：

```bash
vas vocab show                                   # 現在の用語集を表示
vas vocab add "西丸" --alias にしまる --note "ユーザー本人の姓"

export VAS_SLACK_USER_TOKEN=xoxp-...             # Slack ユーザートークン（scope: search:read）
vas vocab import-slack                           # 直近90日分の自分の発言から用語集を生成
```

`VAS_SLACK_USER_TOKEN` は **ユーザートークン**（`xoxp-`）である必要があります（`search.messages` はボットトークンでは使えません）。抽出には `[correct] model` と同じモデルが既定で使われます。

## 開発

```bash
uv pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
VAS_ASR_BACKEND=fake VAS_DATA_DIR=/tmp/vas .venv/bin/vas ingest some.wav   # ASR なしで配管だけ試す
```

## 費用・容量の目安

- 録音: AAC-LC 32kbps mono ≈ 14MB/時 → 14 時間/日で約 200MB/日、約 75GB/年（削除しない方針）。
- 文字起こし: ローカル無料。VAD 後の実発話 3〜4 時間/日なら Apple Silicon で 20〜30 分程度。
- 要約: エピソード別は Haiku 4.5、日次まとめは Opus 5 で月 $5〜15 程度。

## 今後

- iPhone 常時録音アプリ（iOS はバックグラウンドからマイクを再開できない制約があるため別設計）
- Zoom クラウド録画の自動取り込み、Google カレンダー連携でのエピソード命名
- ローカル Web UI、ベクトル検索
