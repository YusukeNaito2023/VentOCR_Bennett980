# CHANGELOG

## 0.1.5 (2026-10-25)
### Results no longer depend on where the app cuts the segments
Comparing the app's current form (CFR segments of 58-67 s) with the 0.8.0 form (direct decoding, exactly 60 s segments) on a 40-minute PB980 video
showed that the differences came from the engine: Pass 1 and Pass 2 run per segment.
- B. BreathMarks at segment boundaries (post_ocr): Pass 1 does not carry the last icon of the previous segment over, so a breath starting on the first
  frame of a segment was missed. post_ocr now looks at the whole BM_Best sequence and adds the _End -> _Start mark where the segment changes
  (column BoundaryMark = 1, BreathMark_raw "... (segment boundary)"). On the 40-minute video: 3 marks added in the current form and 1 in the 0.8.0 form,
  exactly the breaths each form had missed; both forms now have the same breaths
- A. VTE and Ppeak of breaths that Pass 2 could not read (extract_waves): a breath crossing a segment boundary (and a breath added by B) was never read,
  so about 6% of VTE values were empty (42 of 682), and which ones depended on the segment boundaries. They are now read from the whole-video frames
  (Frames/_flat): about 2 frames per breath (the midpoint for VTE, the next breath's first frame for Ppeak). The number of frames and the time are logged
- C. VTE OCR: VTE is an integer, so digit boxes are no longer joined with a decimal point ("41" + "7" became 41.7, a label fragment "5" + "379" became 5.379).
  Boxes lower than 60% of the tallest (fragments of the "VTE" label) are dropped and the rest are joined left to right. The ROI is padded with 30 px of
  background before OCR, because the last digit touches the right edge and was dropped ("417" read as "41")
- Check (2026-09-08 17-19-23.mkv, 40.6 min, 5 workers; current form = app-072 with CFR segments, 0.8.0 form = app-073 9a8d5bc with direct decoding):
  - Both forms now give 685 breaths, all matched; VTE identical in 685/685 breaths; Ppeak identical in 684/685 (one "2" misread in the current form)
  - Compared with 0.1.4: empty VTE 42 -> 2 and empty Ppeak 42 -> 2 in both forms (only the first/last breaths remain); broken VTE (decimals, digits dropped) 17 -> 7
    and 15 -> 7, the remaining 7 being real small volumes (10-65 mL) of stacked breaths
  - A read 86 frames for 43 breaths in 10.8 s (current form; 2.4 s of it loading the OCR model) and 82 frames for 41 breaths in 15.7 s (0.8.0 form; 3.2 s)
  - Alerts still differ between the forms (Fragment 102 / 94, No Sweep 6 / 1): the sweep line is found more often in the cleaner frames of the 0.8.0 form

## 0.1.4 (2026-10-25)
### No Japanese in the engine (English only)
- manifest.json: labels are plain English strings (app's label format): breath_values ("Mode (A / C / S)", "Ppeak (peak pressure)") and outputs.alerts (the {"ja", "en"} labels became the English string)
- Exception and log messages in English ("PB980 screen not found ...", "Detected resolution: 1280x720 (read upscaled to 3840x2160)")
- Code comments and docstrings, diff_viewer.py messages, the .gitignore comment and the Alert section of README.md in English
- The analysis results do not change (only comments, messages, labels and ENGINE_VERSION)
- Entries for earlier versions below are kept as they were (in Japanese)

## 0.1.3 (2026-10-25)
### 出力の宣言（manifest.json の outputs。ENGINE_API.md §8.1、Hamilton 0.2.4 に合わせる）
- waves: Pressure（表示名 Paw、cmH2O、#d62728）・Flow（L/min、#1f77b4）・Volume（mL、#2ca02c）。縦軸は呼吸器の画面の目盛り（P_min〜P_max・F_min〜F_max・V_min〜V_max）
- breath_values: Mode（換気モード）・VTE（mL）・Ppeak（cmH2O）・VTE_raw（表示名 VTE (displayed)。画面の表示は1つ前の呼吸の値）
- alerts: Fragment・Abnormal・No Sweep・Settings Change・Invalid Range・Empty Waveform・Frame Read Error。どれも波形が欠ける・乱れる呼吸なので、
  精度の評価から除く（exclude_from_accuracy）。1つの呼吸に複数のときは「, 」で区切る（例「Abnormal, Fragment」）
- 解析の結果は変わらない（manifest.json と ENGINE_VERSION だけ）

## 0.1.2 (2026-10-21)
### 波形を画面の線のまま書き出す（平滑化・直線の補間を既定で切る）
- 以前は設定に smoothing・imputation が無いと両方 ON とみなし、色で拾えない列を前後から直線で埋め、10px（4K。1280x720 では約 3 画素・約 0.08 秒）の
  移動平均をかけていた。画面の下の端より下に外れた区間（Volume が負・Flow が目盛りの外）や、設定のポップアップで隠れた区間が直線になっていた
- 既定を両方 OFF にし、拾えない点は空のまま残す（アプリは線を途切れて描く）。設定 post_ocr.smoothing・post_ocr.imputation（schema.py・engine_options.json）で入れ直せる
- 確認（VentOCR 0.7.1、0.1.1 と同じ時刻の点どうしで比べて）:
  - 17-19-23.mkv（1280x720、Alert の無い 574 呼吸）: 差の中央値 Pressure 0.01・Flow 0.34・Volume 2.1 mL、95% で Volume 17 mL 以内。
    空になった点は Pressure 0.2%・Flow 0.1%・Volume 1.5%（ほぼすべて 0.1.1 で直線で埋めていた点）
  - Bennett980_Sample.mp4（4K、Alert の無い 57 呼吸）: 差の中央値 Pressure 0・Flow 0.21・Volume 1.3 mL
  - 呼吸の区切り・VTE・Ppeak・Mode・Alert・Frames.csv は同じ。区切りの合わせ込み（Wave_T0・Wave_Trim_*）と、目盛り（P/F/V_min・max）が一部の呼吸で少し変わる

## 0.1.1 (2026-10-21)
### 動画とエンジンが合っているかの確認（VentOCR 0.6.3 から）
- Frames.csv に BM_Best_conf（BM_BestScore を 0〜1 に収めたもの）を追加。manifest.json に "frame_check_columns": ["BM_Best"]
  （アプリは先頭 50 フレームの BM_Best の値を、BM_Best_conf が 0.5 未満なら読めなかったとみなして数え、2 割未満なら確認を出す）
- 先頭 50 フレームでの BM_Best_conf: PB980 の画面は 0.81〜1.00（4K・1280x720）、Hamilton・V600・NAVA・PostNAVA の画面は 0.11 以下
- "frame_check": false は残す（受け口の無い古いアプリが VTE で確かめると、区切りの間でしか読まない VTE は先頭 50 フレームでほぼ空なので、
  正しい PB980 の動画でも確認が出てしまう）。ventocr_min を 0.6.3 に
### PB980 の画面でない動画は解析を失敗にする
- post_ocr で、呼吸の区切りが 0 個で Abnormal のフレームが 9 割以上（NO_SCREEN_ABNORMAL_RATIO）なら、「PB980 の画面が見つかりません」の例外を出す。
  以前は全体が呼吸 1 つの「完了」になっていた
### ログ
- 「Detected resolution」に元の大きさも出す（例「1920x1080（3840x2160 に拡大して読む）」。以前は拡大後の 3840x2160 だけで、結果には影響なし）

## 0.1.0 (2026-10-19)
- 0.1.0（旧 v1.3.3 と中身は同じ。版の付け方を 0.X.Y に改めた）

## 1.3.3 (2026-10-19)
### 解析を速く（結果は実質同じ）
- A. BreathMark のテンプレートマッチングを行列の積に: 同じ大きさどうしの TM_CCOEFF_NORMED は「平均を引いたベクトルどうしの余弦」なので、
  見本 6 枚を平均を引いて正規化した行列にしておき、毎フレーム 1 回の積で 6 つのスコアを出す（cv2.matchTemplate × 6 の 14〜16ms → 0.3〜0.5ms/フレーム）。
  差は 1e-6 程度で、BM_* の小数 4 桁目が数フレームで 1 違うことがある
- B. 捜査線の検出（detect_sweep_x_band）: HSV への変換を 3 つの波形 ROI の行だけにし、列のまとまりを numpy で探す（結果は同じ）
- C. Pass 1 で 1280x720・1920x1080 の全画面を 4K に拡大しない: 捜査線の帯と BreathMark の領域だけを、余白を付けて元のフレームから切り出して拡大する
  （crop_reference。倍率が整数なので位置がそろう。全画面を拡大して切り出したものとの差は一部の画素で最大 1 階調）
- D. OCR のモデル（easyocr.Reader）を区間ごとに作り直さない（_get_reader。アプリのワーカーのプロセスは区間をまたいで使われる）
- E. 波形の抽出で同じフレームを読み直さない（BreathsToCSVs の中で直近 4 枚を覚えておく。終了フレームは捜査線の検出・設定画面の判定・波形の抽出と、
  次の呼吸の「開始フレームの1つ前」で使う）。is_settings_screen_img を追加

### 確認（VentOCR 0.5.0、並列数 5、1.3.2 と比べて）
- 2026-09-08 17-19-23.mkv（1280x720、40.6 分、682 呼吸）: 591.1 → 448.8 秒（−24.1%。OCR −25.9%、波形の抽出 −16.3%）。
  Breaths.csv の区切り・VTE・Ppeak・Mode・Alert・Sweep・Region・Wave_* は同じ（違うのは BM_* のスコアの小数 4 桁目だけ）、
  BreathCSVs は 682 ファイルすべてバイト単位で同じ。Frames.csv の Sweep_X は 24455 フレーム中 10 フレームで違う（最大 1px、片方だけ検出が 3）
- Bennett980_Sample.mp4（4K、CFR veryfast）: 71.3 → 61.1 秒（−14.3%）・96.0 → 87.8 秒（−8.5%）。Breaths.csv・BreathCSVs はすべて同じ

## 1.3.2 (2026-10-18)
### 画面の x を元の動画の画素で書き出す（1.3.1 の不具合）
- 1.3.1 は 1280x720 などを 4K に拡大して読むため、Breaths.csv の Sweep_Start / Sweep_End / Region_X_Start / Region_X_End と
  Wave_Rate を 4K の画素で書いていた。アプリは元の動画の幅で位置を出すので、キャプチャーの走査線が 3 倍右に出ていた
- これらは元の動画の画素に戻して書く（`detect_sweep.source_scale`）。波形 CSV の行も元の動画の 1 画素 = 1 行に間引く
  （拡大した分の行は補間で作ったもの）。Wave_Trim_Head / Wave_Trim_Tail も元の動画の画素の数。4K の動画は変わらない

### フレームごとのペン先の位置（Frames.csv の Sweep_X）
- Pass 1 で毎フレームの捜査線を検出し、元の動画の画素で Sweep_X に書く（検出できないフレームは空）。
  アプリはこれがあるとキャプチャーの走査線・チャートの横軸にこれを使う
- `detect_sweep.detect_sweep_x_img`（画像から検出。HSV への変換を波形の帯だけにした。結果は detect_sweep_x と同じ）

### 新しい画面の最初の呼吸を ROI 左端から切り出す
- 以前は「ROI 左端 + ROI 幅の 7.16%（206px）」からで、これは 1 画面 6 秒（479.5px/秒）の sample.mp4 に合わせた値。
  1 画面 24 秒の動画では 1.7 秒になり、吸気が丸ごと抜けて呼気だけの波形になっていた（2026-09-08 17-19-23.mkv で 105 / 682 呼吸）
- ROI 左端から切り出し、trim_to_breath（_snap_to_breaths）で吸気の始まりへ合わせる。前の呼吸（右端で見切れた呼吸）と時間がつながっていれば、
  左端〜吸気の手前は前の呼吸の呼気の続きとして渡す。trim_to_breath が OFF のときは、描き始めの余白を秒（NEW_SCREEN_LEAD_IN_SEC = 0.43 秒）で決める
- 開始フレームの sweep_x が左端から NEW_SCREEN_MAX_SEC（1 秒）以内で、前の呼吸と同じ画面でない呼吸も同じ扱いにする。
  開始フレームではもう吸気が始まっていて、sweep_x から切り出すと立ち上がりの足元が欠けていた（同じ動画で 12 呼吸）
- 掃引の速さは Sweep_X の進み方から求める（`_sweep_rate_from_frames`）

### engine_api 3
- manifest.json の engine_api を 3 に（ENGINE_API.md §5.1。アプリは全フレームの画像を保存せず、エンジンが cv2.imread で開いたフレームだけを作る）。
  VentOCR 0.5.0 から使える（それより前のアプリでは使えない）
- 前のフレームがあるかを os.path.isfile で確かめる（engine_api 3 では、作られていないフレームの画像は isfile だけが True を返す）。
  フレームはすべて cv2.imread で、必要なもの（呼吸の開始・終了・その1つ前、Pass 2 の中点と終わり）だけを読む
- manifest.json に "frame_check": false（最初の数秒では値を読まないので、先頭 50 フレームの確認から外す）

### 確認
- Bennett980_Sample.mp4（4K）: 1.3.1 と 89 呼吸、区切り・VTE・Ppeak・Mode・Alert・Sweep・目盛りがすべて一致。
  波形が変わったのは 11 呼吸: 新しい画面の直前の見切れた呼吸が次の画面の左端〜吸気の手前（約 0.46 秒）を受け取り、
  新しい画面の最初の呼吸（sweep_x 766〜847 から切り出していたもの）が吸気の足元から始まる
- 2026-09-08 17-19-23.mkv（1280x720）: 682 呼吸、区切り・Alert は 1.3.1 と同じ。呼吸 7・14・21（解析失敗のタグ）が吸気の足元から始まる。
  Sweep_X は 24455 フレーム中 21513 フレームで検出。呼吸 27 は呼気の途中に実際に 2 回目の吸気がある（画面で確認、エンジンの誤りではない）

## 1.3.1 (2026-10-18)
### 1280x720・1920x1080 の動画を解析できるように
- 以前は 3840x2160 以外の動画で、波形の切り出しが 4K の座標のまま行われて落ちていた
  （`ValueError: operands could not be broadcast together with shapes (2877,) (760,)`）
- 3840x2160 以外の 16:9 のフレームは、読み込むときに 3840x2160 に拡大する（`detect_sweep.to_reference_size` / `read_frame`）。
  PB980 の画面をそのまま取り込んだ動画は解像度が違っても配置が同じなので、4K の座標・閾値がそのまま使える。
  以前の 1280x720 用の座標（test.mov）は、2026-09 の取り込みとは位置が合っていなかった

### 呼吸マークの見つけ方をテンプレートマッチングに
- 見本画像（A/C/S × Start/End）との比べ方を、3D 色ヒストグラムの相関から、グレースケールの正規化相互相関（TM_CCOEFF_NORMED）に変えた。
  2026-09 の 1280x720 の取り込みは色合いが違い（背景 BGR 48,24,8 / 見本 42,31,23）、ヒストグラムでは全フレームが 0.85 未満で Abnormal になり、
  吸気のフレームも End と判定されて呼吸が 0 件だった
- Abnormal の閾値 `BREATHMARK_ABNORMAL_SCORE` を 0.85 → 0.65。アイコンがあるフレームは 4K で 0.99 以上・1280x720 で 0.75〜0.86、アイコンが消えたフレームは 0.53

### 捜査線の最小幅を 10px → 7px
- 1280x720 を拡大したフレームでは捜査線の帯が 8〜11px しかなく、半分ほどの呼吸が No Sweep になっていた（4K では 11〜14px）

### 確認
- Bennett980_Sample.mp4（4K）: 1.3.0 と 1.3.1 で 89 呼吸、開始フレーム・VTE・Ppeak・Mode・Alert・Sweep・切り出し範囲がすべて一致
- 2026-09-08 18-30-03.mkv（1280x720）の 60 秒: 1.3.0 は落ちる / 1.3.1 は 21 呼吸、VTE 416〜441（画面の表示 441）、No Sweep 1・Fragment 4

## 1.3.0 (2026-10-11)
### 呼吸の間の抜けを無くす
- 以前は、波形を「開始フレームの sweep_x 〜 終了フレームの sweep_x」で切り出していたため、
  前の呼吸の最後のフレームから開始フレームまでの 1 フレーム分（約 0.1 秒、4K で約 48px）がどの呼吸にも入っていなかった
- パターン4（開始・終了とも sweep_x がある）は「前の呼吸の最後のフレームの sweep_x の次の列」から切り出す
  （前の呼吸で検出済みの値を使うので、検出の回数は増えない）

### 波形の時刻を書き出す
- Breaths.csv に Wave_T0（最初の点の時刻・秒）/ Wave_Rate（px/秒）を追加。終了フレームの sweep_x（ペン先）から逆算
  （アプリの終了側基準と同じ）。終了フレームで見切れた呼吸は開始フレームの sweep_x から
- 波形 CSV の X は 1 からの行番号（以前と同じ）

### 波形の区切りを吸気の始まりへ合わせる
- 1呼吸 = 吸気の始まり 〜 次の吸気の始まり（NAVA v1.3 の「呼吸の範囲への切り詰め」と同じ考え方）。
  Flow の吸気の立ち上がり（4 L/min 超が 0.06 秒以上続き、12 L/min 以上に届き、その前に吸気でない区間が 0.06 秒以上）を見つけ、
  立ち上がりの足元（最大値の 20% を超えた点から下がり続ける所までさかのぼる）の 0.04 秒前を区切りにする（`_snap_to_breaths`）
- この機種は呼吸どうしの波形が時間の上で隙間なくつながっているので、NAVA のように削るのではなく、
  区切りを「元の区切りに最も近い吸気の始まり」へ動かし、点を前後の呼吸の間で移す（データは消えない）。
  前後の呼吸に波形が無い端だけ、吸気の始まりより前と、呼気を伴わない最後の吸気（次の呼吸の吸気の途中）を削る
- 呼吸の Frame / End_Frame（表示の更新による区切り）と OCR の値は変えない。変わるのは波形 CSV と Wave_T0 / Wave_Rate。
  動かした点の数は Breaths.csv の Wave_Trim_Head / Wave_Trim_Tail（正 = 削った / 渡した、負 = 受け取った）
- 区切り直した呼吸の Wave_Rate は、両端の点の時刻から求め直す（受け取った点は元の呼吸の速さで時刻が決まっているため）
- 設定 `post_ocr.trim_to_breath`（既定 ON）で元の動作に戻せる
- Flow の目盛りが求まっていない呼吸（画素のまま）は対象外
- 確認（~/Downloads/980_sample.mp4）: 82 呼吸。呼吸マークの区切りはもともと吸気の始まりに近く、動いたのは中央値 0.06 秒。隣の呼吸との 0.05〜0.5 秒の隙間 56 → 8 か所（残りは画面の右端と次の画面の切り替わり）。掃引の速さを全体で1つとしているため、速さが約 5% 違う呼吸で 0.16 秒の重なりが1か所出る（元からの時刻の求め方の限界）

## 1.2.0 以前
- SERVER_DIFF.md を参照
