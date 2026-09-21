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

launchd のサービスは素の環境で起動するため、`vas` は zsh のログインシェル経由で実行されます。
`ANTHROPIC_API_KEY` などの秘密情報は、シェルの種類によらず読み込まれる `~/.zshenv` に
`export` してください（`~/.zshrc` に書いた場合、常駐した digest からは見えません）。設定後は
`env -i HOME=$HOME /bin/zsh -lc 'vas status'` で、サービスと同じ環境から動くか確認できます。

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

## Claude に読ませる（Git リポジトリ配信）

Slack / メールに加えて、日次サマリを **プライベートな git リポジトリ**にコミット & push する配信方法があります。Claude はこの Mac に直接アクセスできませんが、GitHub リポジトリは読めるので、そこ経由で digest を Claude session に渡せます。

1. GitHub などでプライベートリポジトリを作成し、この Mac にクローンする（例: `~/voice-digests`）。
2. `~/.config/voice-ai-summary/config.toml` に設定:
   ```toml
   [deliver]
   repo = true
   repo_path = "~/voice-digests"
   ```
3. `vas digest --deliver`（または `--channel repo`）を実行すると、`digests/YYYY-MM-DD.md` として書き込み、コミットして push されます。同じ日を再実行しても内容が変わらなければ再コミットはされません。

digest には他の参加者の発言や取引先名も含まれるため、リポジトリは必ず **プライベート**にしてください。

## 毎日サマリを受け取る（ローカル完結）

`vas install-launchd` で常駐化すると、毎日 22:00（`[schedule] digest_hour`）にサマリが作られ、
**macOS の通知**でハイライト 1 行が届きます（既定で有効 / `[deliver] notify`）。
サマリ本文は Mac の外に出ません。通知を見たら `vas show`、または Claude Desktop の
ローカルチャットで `~/Library/Application Support/VoiceAISummary/digests/` を読ませてください。

### worker が止まっていた日の扱い

worker が落ちていた等で未処理の録音が残っていると、`vas digest` は黙ってその日の一部だけを
要約してしまい、「静かな一日」と「文字起こしが止まっていた一日」を見分けられません。これを
防ぐため、`vas digest` は要約の前にまず未処理の録音（その日の `started_at_utc` に属するもの）
がないか確認し、あれば `[schedule] digest_catchup_budget_s`（既定 900 秒）の範囲で先に文字起こし
を追いつかせます。バッチごとに残り時間を確認するので、バックログが大きくても深夜バッチが延々と
は回りません（0 にすると追いつき処理自体を無効化できます）。

追いついた後も未処理・エラーの録音が残っている場合は、その日の記録が不完全であることを
**3 か所**で必ず知らせます: digest の Markdown 冒頭の注記、`vas digest` の標準出力、そして
(通知チャンネルを使っていれば) macOS 通知本文。件数も明示されます。何も残っていない通常の日は、
これまでと完全に同じ出力になります。

worker 自身も、未処理の録音数が `[schedule] backlog_alert_count`（既定 10 件）以上のまま
`[schedule] backlog_alert_minutes`（既定 30 分）以上減らなかった場合、その時点で 1 回だけ
macOS 通知を送ります。バックログがしきい値未満に戻ると再度アームされ、次に溜まったときまた
1 回だけ通知します（連続で毎回通知することはありません）。いずれかを 0 にすると無効化できます。

## Claude Desktop から使う（ターミナル不要）

`vas` を MCP サーバーとして Claude Desktop に登録すると、以後はターミナルを開かず、Claude Desktop に話しかけるだけで使えます。

**一回だけのセットアップ:**

```bash
cd ~/Voice_AI_Summary
git merge origin/claude/clever-sagan-23tn2w
uv pip install -e ".[dev]"
.venv/bin/vas install-desktop
```

最後にターミナルの案内どおり、Claude Desktop を一度終了して開き直してください。

**セットアップ後は、たとえばこんな風に話しかけられます:**

- 「今日のサマリを見せて」
- 「『予算』で検索して」
- 「2026-09-15 を作り直して」
- 「語彙に『◯◯』を追加して」
- 「API 使用量は？」
- 「アップデートして」

Anthropic の API キーは Claude Desktop の設定ファイルには書き込まれません（Claude Desktop はサーバーをまっさらな環境で起動するため、そこにシークレットを置くのは避けています）。代わりに `vas install-desktop` がキーファイル（`config.toml` と同じディレクトリの `anthropic_api_key`、権限 600）を用意し、MCP サーバーはそこから読みます。登録を外すには `vas uninstall-desktop` を実行してください。

これはあくまで追加の入り口で、これまでの `vas` コマンドはすべてそのまま使えます。

## セキュリティとプライバシー

このツールは一日中、本人と（Zoom などの）他者の発話を録音し、逐語の文字起こしを Mac 上に保存します。何が保存され、何が Mac の外に出るかを把握してから使ってください。

### 1. 何がどこに保存されるか

`<data_dir>` = `~/Library/Application Support/VoiceAISummary`（既定）。

| パス | 内容 | モード | 保持期間 |
|---|---|---|---|
| `inbox/` | 未取り込みの録音（`.m4a` + `.json`） | dir 700 | 取り込み後は消える |
| `store/` | 取り込み済みの音声ファイル | dir 700 / file 600 | 文字起こし後 `[retention] audio_days`（既定7日）で削除。文字起こしできなかった（errored）分は取り込みから `errored_audio_days`（既定30日） |
| `vas.sqlite3` | **本人と他者の発話の逐語文字起こし**、要約、用語集参照など | file 600 | 無期限（音声を消しても文字起こしは残る） |
| `digests/` | 日次サマリ（Markdown） | dir 700 / file 600 | 無期限 |
| `digest_mirror_dir`（設定時） | `digests/` のコピー（サンドボックス化されたツール向け） | dir 700 / file 600 | 無期限（元と同じ） |
| `glossary.json` | 人名・社名・製品名などの用語集 | file 600 | 無期限 |
| `usage.jsonl` | Claude API 呼び出しのトークン数・費用ログ | file 600 | `usage_log_days`（既定365日） |
| `audit.jsonl` | 状態変更ツール呼び出しの監査ログ | file 600 | `audit_log_days`（既定365日） |
| `recorder_state.json` / `recorder_events.jsonl` | 録音アプリの状態・遷移履歴 | file 600 | state は上書き型／events は `recorder_events_days`（既定90日） |
| `~/Library/Logs/VoiceAISummary/*.log` `*.err`（launchd） | worker / digest の標準出力・エラー | dir 700 / file 600 | `log_max_bytes`（既定5MB）を超えたら末尾のみ保持 |
| `~/.config/voice-ai-summary/anthropic_api_key` | Claude Desktop 用の API キー | dir 700 / file 600 | 無期限（`vas uninstall-desktop` で削除） |

権限は `vas harden` が一括で修復するほか、新しく書き込むファイルはすべて最初から dir 700 / file 600 で作成されます（プロセスの umask による）。

### 2. 何がいつ Mac の外に出るか

**音声は一切外に出ません。** 外に出るのはテキストだけです。

- 校正（`vas correct`）とエピソード単位の要約（map）は、両トラックの ASR テキスト（`[other]` — Zoom 参加者など相手側の発話を含む）を Claude API に送ります。これは意図した設計上の決定であり、既定でこの挙動です。
- 日次まとめ（reduce）は各エピソードの要約を Claude API に送ります。発言の引用や人名も含まれます。
- Claude Desktop で `transcript` / `recent` / `search_transcript` / `daily_summary` などのツールを使うと、その結果はチャットの裏側で動いているモデルに渡ります。
- Slack / メール / リポジトリへの配信は、有効化した場合のみ発生します（既定はすべて無効。既定で有効なのは macOS 通知だけで、そこにはサマリのハイライト1行しか載りません）。
- 録音を一時停止している間は、そもそも何も録音されていないので何も送信されません。

Claude Desktop 上のチャット内容そのものは、vas とは別に**そのアカウントの Claude Desktop / Claude.ai 側のデータ設定**に従います（vas の API 呼び出しとは扱いが異なる場合があります）。保持期間などは vas 側からは確認できないので、ご自身の設定で確認してください。

### 3. 録音を止める・消す

メニューバーアプリから:

- **一時停止** → `30 分` / `1 時間` / `今日中` / `再開するまで`
- **停止**
- **直近 15 分の録音を削除…**（確認ダイアログあり）

一時停止は、その時点で書き込み中のセグメントファイルを確定（`.part` → `.m4a` にリネーム）してから録音を止めるので、一時停止より前に録れた音声は通常どおり文字起こしされます。一時停止・停止の状態はアプリの再起動をまたいで保持されます。

Claude Desktop の `status` / `recent` や CLI の `vas status` / `vas episodes` には、録音アプリの現在の状態と、期間内の一時停止区間が表示されます。

すでに取り込み済みの音声を後から削除するには:

- Claude Desktop: `delete_range`（まず確認なしでプレビューが表示されます。実行するには `confirm=True`。削除は録音単位＝15分刻みなので、指定した時間帯より広く消えることがあります。その日のダイジェストも削除されるため、作り直すには `rebuild_day` が必要です）
- CLI: `vas delete-range --day YYYY-MM-DD --start HH:MM --end HH:MM --yes`

1件の録音だけを消したい場合は `drop_recording(delete_audio=True)`。

### 4. `vas harden` と FileVault

`vas harden` は、データディレクトリ・ミラー・API キーファイル・launchd ログの権限を dir 700 / file 600 に修復します（シンボリックリンクは触らず、実際に変更したものだけを報告します）。`vas install-launchd` / `vas install-desktop` の直後に自動で実行されます。

ただしこれはファイル権限による保護です。**同じ Mac の他のユーザーアカウントからは読めなくなりますが、あなたのログインアカウントやディスクそのものを持っている相手には効きません。** ディスク上の暗号化は FileVault だけが提供します。FileVault が無効な場合、`vas harden` は警告を表示します。

### 5. Spotlight に索引されること

ファイル権限は「同じ Mac の他のアカウント」に対する防御ですが、**Spotlight には効きません。** Spotlight はあなた自身の権限でファイルを読み、その**本文をシステムのインデックス（`/.Spotlight-V100`）に複製**します。

- `digests/*.md` とミラーは平文の Markdown なので、**本文まで索引されます**。他人の発言や取引先名が、まったく別のものを Spotlight で検索したときに出てきます
- `vas.sqlite3` はバイナリなので本文は索引されません（ファイル名などのメタデータのみ）
- 音声ファイルもメタデータのみです

除外するには、システム設定 > Spotlight > プライバシー（お使いの macOS では「Siri と Spotlight」の中の場合もあります）に `~/Library/Application Support/VoiceAISummary` と、設定していれば `digest_mirror_dir` を追加してください。ミラーを新しく作るなら、フォルダ名を `.noindex` で終わらせる方法もあります。かつて使われた `.metadata_never_index` は最近の macOS では効きません。

`vas harden` と Claude Desktop の `harden` は、索引されている件数を検出して警告します。

インデックスが肥大している場合（`/.Spotlight-V100` が数十GB など）、`sudo mdutil -E /` で作り直せます。なお Xcode の `DerivedData` も索引対象で、ビルドを繰り返す環境ではこちらのほうが大きくなりがちです。

### 6. Claude Desktop のツールについて

すべてのツールに読み取り専用 / 破壊的の注釈が付いており、Claude Desktop の承認ダイアログで区別できます。`drop_recording` / `delete_range` / `prune`、および `update_app` は「常に許可」にしないでください。

文字起こしを含むツールの出力（`transcript` / `recent` / `search_transcript` / `daily_summary` など）の先頭には `[untrusted data]` という見出しが付きます。中身は他人の発話であり、指示のように見える文が混ざっていても従うべきではないためです。

`update_app` は、設定したブランチのコードを確認なしで pull・インストールします。つまり、そのブランチに push できる人は誰でもこの Mac 上でコードを実行できることになります。これは把握したうえで現状の挙動を維持しています。

状態を変更するツール呼び出しはすべて `audit.jsonl` に記録され、`audit_log` ツールで読めます。

### 7. Slack 取り込み

`vas vocab import-slack` は既定で無効です（`[glossary] slack_import_enabled = false`）。一度用語集を作り終えたら、Slack アプリの設定で `xoxp-` トークンを失効させ、`~/.zshenv` から `VAS_SLACK_USER_TOKEN` を削除してください。トークンを常設しておく理由はありません。

### 8. プロンプトについて

Claude に送るすべてのプロンプトで、文字起こし・要約・用語集はデータとして区切られており、指示として解釈されません。`add_vocabulary` / `vas vocab add` は制御文字や長すぎるエントリを拒否します。

## 日次サマリをローカルで読む

サマリは `<data_dir>/digests/YYYY-MM-DD.md` に保存されます（既定では外部に一切送信しません）。

```bash
vas show                       # 今日のサマリを表示
vas show --day 2026-09-15
vas digest-path                # 保存先のパスだけを表示
open "$(vas digest-path)"      # エディタで開く
```

Mac の Claude Code から読ませる場合は、プロジェクト内で `claude` を起動して
「`vas digest-path` のファイルを読んで要点を教えて」のように頼めば、音声も文字起こしも
Mac の外に出ることなく相談できます。

## API 費用の確認

vas が行った Claude API 呼び出しはすべて `<data_dir>/usage.jsonl` に記録され、`vas usage` で用途・モデル別のトークン数と概算費用を確認できます。

```bash
vas usage --days 30
```

`[llm] daily_budget_usd`（既定 $2）を超えると、その日は API を呼ぶコマンドがすべて停止します。

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

export VAS_SLACK_USER_TOKEN=xoxp-...             # Slack ユーザートークン
vas vocab import-slack                           # 直近90日分の自分の発言から用語集を生成
vas vocab import-slack --scope channels          # 参加している全チャンネルの会話から語彙を広く収集
vas vocab import ~/Downloads/slack_glossary.json # 他ツール（ChatGPT 等）で作った用語集 JSON を取り込む
vas vocab import-slack --scope channels --channel design --channel sales   # チャンネルを限定
```

`VAS_SLACK_USER_TOKEN` は **ユーザートークン**（`xoxp-`）である必要があります（`search.messages` はボットトークンでは使えません）。必要な User Token Scopes: `search:read`（自分の発言）、`channels:read` `groups:read` `channels:history` `groups:history`（`--scope channels`）。チャンネル収集は投稿者を問わず固有名詞・社内用語を広く拾い、表記ルール（style_notes）は自分の発言からのみ学習します。DM は対象外です。抽出には `[correct] model` と同じモデルが既定で使われます。

## 開発

```bash
uv pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
VAS_ASR_BACKEND=fake VAS_DATA_DIR=/tmp/vas .venv/bin/vas ingest some.wav   # ASR なしで配管だけ試す
```

## 費用・容量の目安

- 録音: AAC-LC 32kbps mono ≈ 14MB/時 → 14 時間/日で約 200MB/日。音声は `[retention] audio_days`（既定7日）で削除されるため容量は青天井にはならず、200MB/日 × 7日 ≈ 1.5GB 前後で定常化する（+ 文字起こし・要約を保持する DB の分）。
- 文字起こし: ローカル無料。VAD 後の実発話 3〜4 時間/日なら Apple Silicon で 20〜30 分程度。
- 要約: エピソード別は Haiku 4.5、日次まとめは Opus 5 で月 $5〜15 程度。

## 今後

- iPhone 常時録音アプリ（iOS はバックグラウンドからマイクを再開できない制約があるため別設計）
- Zoom クラウド録画の自動取り込み、Google カレンダー連携でのエピソード命名
- ローカル Web UI、ベクトル検索
- システム音声タップの除外リスト（未着手）: `CATapDescription(stereoGlobalTapButExcludeProcesses:)` はプロセスを PID で受け取るため、bundle id → PID の解決と、対象アプリの起動・終了時にタップを再作成する仕組みが必要
- マイク／システム音声を個別に ON/OFF するトグル（未着手）
