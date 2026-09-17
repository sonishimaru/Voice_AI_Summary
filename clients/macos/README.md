# VoiceRecorder (macOS メニューバー録音アプリ)

常時起動のパーソナル録音ツールです。マイク（自分の声）とシステムオーディオ（Zoom/Meet/Teamsなど相手の声、動画の音声など）を **別々の2トラック** として、15分ごとにローテーションするAACファイルに書き出し、インボックスフォルダに保存します。あとで別プロセスのPythonワーカーがこれらを文字起こしします。2トラックに分けているのは「マイク=自分」「システム=相手」を区別するためで、話者分離（diarization）は不要という設計です。

## 前提条件

- macOS 14.4 (Sonoma) 以降
- Xcode 15.3 以降（Swift 5.9 以降）
- [XcodeGen](https://github.com/yonaskolb/XcodeGen)

```sh
brew install xcodegen
```

## ビルド手順

`clients/macos/` ディレクトリで実行してください。

```sh
cd clients/macos
xcodegen generate
open VoiceRecorder.xcodeproj
```

Xcodeが開いたら:

1. `VoiceRecorder` ターゲットを選択し、「Signing & Capabilities」タブを開く
2. 「Team」に自分のApple ID（Personal Team可）を設定する
3. `⌘R` でビルド&実行

初回起動時、メニューバーに波形アイコンが表示されます（Dockアイコンは出ません＝`LSUIElement`）。

コードを更新（`git pull`）したあとは、**Xcodeを終了してから** `xcodegen generate` をやり直してください。追加されたソースファイルや `Info.plist` のキーは、プロジェクトを再生成しないと反映されません。Xcodeでプロジェクトを開いたまま再生成すると、ビルド中にファイルが書き換わったと判断されてエラーになることがあります。

> `Entitlements file "VoiceRecorder.entitlements" was modified during the build` と出たとき: ビルドの最中に誰か（Xcode自身の「Signing & Capabilities」タブ、`xcodegen generate`、`git pull`、同期フォルダなど）がこのファイルに触っています。Xcodeを終了 → `xcodegen generate` → Xcodeを開く → Product > Clean Build Folder（`⇧⌘K`）→ もう一度ビルド、で解消します。再発する場合は `git status` を見てください。`VoiceRecorder.entitlements` が変更扱いになっていれば、Xcodeが書き換えた内容がわかります。

## 求められる権限

起動直後、実際に録音を開始しようとしたタイミングで、macOSが以下の2つの許可を求めてきます:

- **マイク** — `NSMicrophoneUsageDescription`。自分の声を録るために必要です。
- **システムオーディオ録音**（`NSAudioCaptureUsageDescription`。「マイク」とは別の許可項目です） — Zoom/Meet/Teamsなど相手の声や動画の音声など、Macが再生している音全体を録るために必要です。

**画面収録（Screen Recording）の許可は不要です。** このアプリはScreenCaptureKitを一切使っていません。ScreenCaptureKit経由のシステムオーディオ録音は画面収録権限を使い回すため、月次で再許可を求められる仕様になっており、常時起動ツールには不向きです。代わりにmacOS 14.2以降のCore Audio「プロセスタップ」API（`CATapDescription` / `AudioHardwareCreateProcessTap` / アグリゲートデバイス経由のIOProc）を使い、システム全体のオーディオ出力を1本のモノラルミックスとしてタップしています。プライベートなTCC APIは使用していません。

許可はシステム設定 > プライバシーとセキュリティ > 「マイク」および「システムオーディオ録音」から後で確認・変更できます。

## ファイルの保存先とローテーション

デフォルトの保存先:

```
~/Library/Application Support/VoiceAISummary/inbox/
```

（`Settings.swift` の `UserDefaults` キーで変更可能）

15分（デフォルト、`rotationMinutes`で変更可）ごとに、マイク・システムオーディオそれぞれのファイルを閉じて新しいファイルを開始します。ファイル名は以下の形式です:

```
{source}_{device}_{YYYYMMDDTHHMMSSZ}.m4a
```

- `source`: `mac_mic`（マイク） または `mac_system`（システムオーディオ）
- `device`: `Host.current().localizedName` を小文字化しASCII英数字とハイフンのみに変換したもの
- タイムスタンプ: そのファイルの録音開始時刻（UTC）

各音声ファイルの隣に、同じファイル名（拡張子だけ`.json`）でメタデータのサイドカーファイルが書き出されます（`started_at_utc`、`tz_offset`、`sample_rate`、`channels`、`codec`、`app_version`など）。

書き込み中は `.part` という拡張子を付けたファイルに書き込み、ファイルを閉じたタイミングで最終的なファイル名にリネームします。これにより、PythonワーカーがまだAAC書き込み中の中途半端なファイルを誤って読みに行くことはありません。

## メニュー項目

- 状態表示: 「録音中 HH:MM〜」/「一時停止中 — HH:MM に再開（残り N 分）」/「一時停止中 — 手動で再開するまで」/「停止中」
- 直前の操作結果（削除の実行結果など）があれば、状態表示の下にもう1行表示されます
- **一時停止**（録音中のみ表示） — サブメニューから停止時間を選びます:
  - `30 分` / `1 時間` — その時間が経過したら自動的に再開します
  - `今日中` — その日の24:00（次の日の00:00）まで一時停止し、日付が変わったタイミングで自動的に再開します
  - `再開するまで` — 期限を設けず、手動で「再開」を選ぶまで一時停止します
  - 一時停止は**現在のセグメントファイルを確定（`.part` → `.m4a` にリネーム）してから**録音パイプラインを止めます。再開すると新しいセグメントファイルから録音が始まるため、一時停止の前後の音声が1つのファイルに継ぎ目なく混ざることはありません
  - 一時停止の状態（いつまで、など）はアプリの再起動をまたいで保持されます。一時停止中にQuitして再度起動しても、録音は自動的に再開しません（時間指定していた場合は、その時刻を過ぎていれば起動時に自動再開します）
  - 時間指定の一時停止から自動再開すると、通知（`UNUserNotificationCenter`、失敗時は`osascript`にフォールバック）で「録音を再開しました」とお知らせします
- **再開**（一時停止中のみ表示） / **開始**（停止中のみ表示）
- **停止** — 現在のセグメントファイルを確定して録音を完全に停止します。次に「開始」を押すまで（またはアプリ再起動時に自動起動する設定の場合はその時）録音は始まりません
- **直近 15 分の録音を削除…** — プライバシー保護機能です。確認ダイアログを表示したうえで、(1) 録音中であれば現在開いている（まだ確定していない）セグメントファイルを両トラックとも削除し、即座に新しいセグメントとして録音を継続し、(2) さらにインボックス内の、直近15分以内に開始された確定済み（ローテーション済み）ファイルも探して削除します。**すでにPythonワーカー側で取り込み（ingest）済みの音声はこの操作だけでは消えません** — その分はClaude Desktop側で`delete_range`を使って削除してください
- ログイン時に起動 — `SMAppService` によるログイン時自動起動のON/OFF
- inbox フォルダを開く — 保存先フォルダをFinderで開く
- 終了 — アプリを終了（終了前に現在のセグメントファイルを同期的に確定します）

## 状態ファイル（外部から録音状態を確認する）

メニューバーを覗く以外に、このアプリが実際に録音中かどうかを外部（Pythonワーカーや監視スクリプトなど）から確認する手段が2つあります。どちらも `stateDirPath` の`UserDefaults`キーで変更可能で、デフォルトではインボックスフォルダの**親ディレクトリ**（`~/Library/Application Support/VoiceAISummary/`）に置かれます。

- `recorder_state.json` — 最新の状態1件だけを保持する上書き型のファイルです。録音中は60秒ごとに更新される（ハートビート）ほか、状態が変わるたびに即座に更新されます。
- `recorder_events.jsonl` — 状態が変わるたびに1行（1つのJSONオブジェクト）が追記されるログです。一時停止・再開・スリープ・削除など、すべての遷移の履歴を追えます。

両ファイルとも所有者のみ読み書き可能（`0600`）です。`recorder_state.json` の例:

```json
{
  "schema": 1,
  "state": "paused",
  "since": "2026-09-17T01:23:45Z",
  "resume_at": "2026-09-17T02:23:45Z",
  "reason": "user",
  "pid": 12345,
  "app_version": "0.2.0",
  "updated_at": "2026-09-17T01:23:45Z"
}
```

- `state`: `"recording"` / `"paused"` / `"stopped"` のいずれか
- `since`: 現在の`state`になった時刻（UTC）
- `resume_at`: 時間指定の一時停止であればその自動再開予定時刻（UTC）、それ以外は`null`
- `reason`: この更新のきっかけ（`"user"` / `"launch"` / `"wake"` / `"sleep"` / `"retry"` / `"device-change"` / `"timer"` / `"heartbeat"`など）
- `pid`: このプロセスのPID
- `minutes` / `files`: `recorder_events.jsonl`の`"delete_recent"`イベントにのみ付与される、削除対象の分数と削除したファイル数

## 既知の制限

- **App Sandboxを無効化しています**（`VoiceRecorder.entitlements` で `com.apple.security.app-sandbox: false`）。Core Audioのプロセスタップ／アグリゲートデバイス作成はまだ新しいAPIで、サンドボックス下での動作保証や必要なentitlementが明確でないため、シンプルに動かすことを優先してサンドボックスをオフにしています。App Store配布は想定していません。
- macOS 14.4 (Sonoma) 以降が必須です（Core Audioプロセスタップは14.2で追加されましたが、本アプリは14.4を最低ターゲットにしています）。
- AirPodsの接続/切断など、デフォルトの入出力デバイスが切り替わるタイミングでは、録音パイプラインを一度止めて再構築します（音声が数秒途切れることがあります）。書き込み中のファイル自体は失われません。
- Mac本体がスリープすると、録音中であれば一度停止（現在のセグメントを確定）します。これは一時停止とは別扱いで、状態ファイルには`reason: "sleep"`で記録されます。復帰時は、スリープ前の状態（録音中/一時停止中/停止中）に応じて自動的に元の状態へ戻ります — 録音中なら新しいセグメントで再開、時間指定の一時停止で予定時刻をすでに過ぎていれば自動再開（通知あり）、まだ先なら一時停止を継続します。
- 本コードはLinux環境で書かれ、実機のXcodeでビルド・検証されていません。ソースコード中の `// NOTE:` コメントは、実機での動作確認が特に必要な箇所（Core AudioのプロパティセレクタやCATapDescriptionのAPI詳細など）を示しています。

## トラブルシューティング

- **ファイルが全く生成されない** → システム設定 > プライバシーとセキュリティ を開き、「マイク」と「システムオーディオ録音」の両方で本アプリが許可されているか確認してください。一度拒否すると自動では再度聞かれないため、ここで手動でONにする必要があります。
- **システムオーディオのファイルだけ生成されない** → 上記の「システムオーディオ録音」権限が入っているか、また `Console.app` で `com.voiceaisummary.recorder` / カテゴリ `system-audio` のログにCore Audioのエラー（`OSStatus`）が出ていないか確認してください。
- **ビルドが `xcodegen generate` の時点で失敗する** → `project.yml` のYAML構文エラーの可能性があります。`xcodegen dump` で解釈結果を確認してください。
- **署名エラーで実行できない** → Xcodeの「Signing & Capabilities」でTeamを設定してください（`project.yml` では意図的に固定していません）。
