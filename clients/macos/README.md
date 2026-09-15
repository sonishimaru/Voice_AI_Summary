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

- 状態表示: 「Recording since HH:MM」/「Paused」/「Stopped」
- Pause / Resume — 一時停止・再開（開いている現在のセグメントファイルはそのまま保持されます）
- Delete last 15 minutes — 直前の最小限のプライバシー保護機能です。現在開いている（まだ確定していない）セグメントファイルを両トラックとも削除し、即座に新しいセグメントとして録音を継続します
- Launch at Login — `SMAppService` によるログイン時自動起動のON/OFF
- Open inbox folder — 保存先フォルダをFinderで開く
- Quit — アプリを終了

## 既知の制限

- **App Sandboxを無効化しています**（`VoiceRecorder.entitlements` で `com.apple.security.app-sandbox: false`）。Core Audioのプロセスタップ／アグリゲートデバイス作成はまだ新しいAPIで、サンドボックス下での動作保証や必要なentitlementが明確でないため、シンプルに動かすことを優先してサンドボックスをオフにしています。App Store配布は想定していません。
- macOS 14.4 (Sonoma) 以降が必須です（Core Audioプロセスタップは14.2で追加されましたが、本アプリは14.4を最低ターゲットにしています）。
- AirPodsの接続/切断など、デフォルトの入出力デバイスが切り替わるタイミングでは、録音パイプラインを一度止めて再構築します（音声が数秒途切れることがあります）。書き込み中のファイル自体は失われません。
- Mac本体がスリープすると録音は一度停止（現在のセグメントを確定）し、復帰時に自動的に新しいセグメントで再開します。
- 本コードはLinux環境で書かれ、実機のXcodeでビルド・検証されていません。ソースコード中の `// NOTE:` コメントは、実機での動作確認が特に必要な箇所（Core AudioのプロパティセレクタやCATapDescriptionのAPI詳細など）を示しています。

## トラブルシューティング

- **ファイルが全く生成されない** → システム設定 > プライバシーとセキュリティ を開き、「マイク」と「システムオーディオ録音」の両方で本アプリが許可されているか確認してください。一度拒否すると自動では再度聞かれないため、ここで手動でONにする必要があります。
- **システムオーディオのファイルだけ生成されない** → 上記の「システムオーディオ録音」権限が入っているか、また `Console.app` で `com.voiceaisummary.recorder` / カテゴリ `system-audio` のログにCore Audioのエラー（`OSStatus`）が出ていないか確認してください。
- **ビルドが `xcodegen generate` の時点で失敗する** → `project.yml` のYAML構文エラーの可能性があります。`xcodegen dump` で解釈結果を確認してください。
- **署名エラーで実行できない** → Xcodeの「Signing & Capabilities」でTeamを設定してください（`project.yml` では意図的に固定していません）。
