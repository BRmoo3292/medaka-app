#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include <RTClib.h>
#include <Preferences.h>

// WiFi設定
const char* ssid = "TP-Link_6888";
const char* password = "33629254";

// サーボ設定
Servo servo1;  // 9時、12時、16時に回転
Servo servo2;  // 12時のみ回転

const int SERVO1_PIN = 13;  // GPIO13
const int SERVO2_PIN = 14;  // GPIO14
const int LED_PIN = 2;      // 内蔵LED

// RTC
RTC_DS3231 rtc;

// 設定保存用
Preferences preferences;

// 餌やり設定
struct FeedingConfig {
  int servo1_rotations = 3;  // サーボ1の回転回数
  int servo2_rotations = 3;  // サーボ2の回転回数
  int feed_hour_1 = 9;       // 9時
  int feed_hour_2 = 12;      // 12時
  int feed_hour_3 = 16;      // 16時
};

FeedingConfig config;

// 餌やり記録（最新10件）
struct FeedingLog {
  String timestamp;
  String servo;
  String type;  // "auto" or "manual"
};

FeedingLog logs[10];
int logIndex = 0;

// 餌やり完了フラグ
bool fed_9am = false;
bool fed_12pm_servo1 = false;
bool fed_12pm_servo2 = false;
bool fed_4pm = false;
int lastDay = -1;

WebServer server(80);

void setup() {
  Serial.begin(115200);
  
  // LED初期化
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // サーボ初期化
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  servo1.write(0);
  servo2.write(0);
  
  // I2C初期化（DS3231用）
  Wire.begin(21, 22);  // SDA=GPIO21, SCL=GPIO22
  
  // RTC初期化
  if (!rtc.begin()) {
    Serial.println("RTCが見つかりません！");
    while (1) {
      digitalWrite(LED_PIN, HIGH);
      delay(100);
      digitalWrite(LED_PIN, LOW);
      delay(100);
    }
  }
  
  // RTCの時刻確認
  if (rtc.lostPower()) {
    Serial.println("RTCの電源が失われていました。時刻を設定してください。");
    // コンパイル時刻で初期設定（後でWebから変更可能）
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  }
  
  // 設定読み込み
  preferences.begin("feeder", false);
  config.servo1_rotations = preferences.getInt("s1_rot", 3);
  config.servo2_rotations = preferences.getInt("s2_rot", 3);
  config.feed_hour_1 = preferences.getInt("hour1", 9);
  config.feed_hour_2 = preferences.getInt("hour2", 12);
  config.feed_hour_3 = preferences.getInt("hour3", 16);
  
  // WiFi接続
  WiFi.begin(ssid, password);
  Serial.print("WiFi接続中");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi接続成功!");
    Serial.print("IPアドレス: ");
    Serial.println(WiFi.localIP());
    Serial.println("このIPアドレスをブラウザで開いてください");
    
    // 成功を示すLED点滅
    for(int i = 0; i < 3; i++) {
      digitalWrite(LED_PIN, HIGH);
      delay(200);
      digitalWrite(LED_PIN, LOW);
      delay(200);
    }
  } else {
    Serial.println("\nWiFi接続失敗。APモードで起動します。");
    WiFi.softAP("ESP32_Feeder", "12345678");
    Serial.print("AP IPアドレス: ");
    Serial.println(WiFi.softAPIP());
  }
  
  // Webサーバー設定
  server.on("/", handleRoot);
  server.on("/feed", handleManualFeed);
  server.on("/logs", handleLogs);
  server.on("/config", handleConfig);
  server.on("/settime", handleSetTime);
  server.on("/status", handleStatus);
  server.begin();
  
  Serial.println("Webサーバー起動完了");
  printCurrentTime();
}

void loop() {
  server.handleClient();
  
  DateTime now = rtc.now();
  
  // 日付が変わったらフラグリセット
  if (now.day() != lastDay) {
    fed_9am = false;
    fed_12pm_servo1 = false;
    fed_12pm_servo2 = false;
    fed_4pm = false;
    lastDay = now.day();
    Serial.println("日付が変わりました。餌やりフラグをリセット。");
  }
  
  // 9時の餌やり（サーボ1）
  if (!fed_9am && now.hour() == config.feed_hour_1 && now.minute() == 0) {
    Serial.println("9時 - サーボ1で自動餌やり");
    feedServo(servo1, config.servo1_rotations, "サーボ1", "自動(9時)");
    fed_9am = true;
  }
  
  // 12時の餌やり（サーボ1とサーボ2）
  if (!fed_12pm_servo1 && now.hour() == config.feed_hour_2 && now.minute() == 0) {
    Serial.println("12時 - サーボ1で自動餌やり");
    feedServo(servo1, config.servo1_rotations, "サーボ1", "自動(12時)");
    fed_12pm_servo1 = true;
    delay(2000);
  }
  
  if (!fed_12pm_servo2 && now.hour() == config.feed_hour_2 && now.minute() == 0) {
    Serial.println("12時 - サーボ2で自動餌やり");
    feedServo(servo2, config.servo2_rotations, "サーボ2", "自動(12時)");
    fed_12pm_servo2 = true;
  }
  
  // 16時の餌やり（サーボ1）
  if (!fed_4pm && now.hour() == config.feed_hour_3 && now.minute() == 0) {
    Serial.println("16時 - サーボ1で自動餌やり");
    feedServo(servo1, config.servo1_rotations, "サーボ1", "自動(16時)");
    fed_4pm = true;
  }
  
static unsigned long lastCheck = 0;
if (millis() - lastCheck >= 1000) {
  lastCheck = millis();
  // 時刻チェック処理をここに移動
}
}

// サーボ回転関数
void feedServo(Servo &servo, int rotations, String servoName, String feedType) {
  digitalWrite(LED_PIN, HIGH);
  
  for (int i = 0; i < rotations; i++) {
    Serial.printf("  %s 回転 %d/%d\n", servoName.c_str(), i + 1, rotations);
    servo.write(180);
    delay(1000);
    servo.write(0);
    delay(1000);
  }
  
  digitalWrite(LED_PIN, LOW);
  
  // ログに記録
  addLog(servoName, feedType);
  
  Serial.printf("  %s 完了\n", servoName.c_str());
}

// ログ追加
void addLog(String servo, String type) {
  DateTime now = rtc.now();
  char timeStr[20];
  sprintf(timeStr, "%04d/%02d/%02d %02d:%02d", 
          now.year(), now.month(), now.day(), now.hour(), now.minute());
  
  logs[logIndex].timestamp = String(timeStr);
  logs[logIndex].servo = servo;
  logs[logIndex].type = type;
  
  logIndex = (logIndex + 1) % 10;
}

// 現在時刻表示
void printCurrentTime() {
  DateTime now = rtc.now();
  Serial.printf("現在時刻: %04d/%02d/%02d %02d:%02d:%02d\n",
                now.year(), now.month(), now.day(),
                now.hour(), now.minute(), now.second());
}

// ===== Webハンドラ =====

// メインページ
void handleRoot() {
  DateTime now = rtc.now();
  
  String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🐟 自動餌やりシステム</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 20px;
    }
    .container {
      max-width: 600px;
      margin: 0 auto;
      background: white;
      border-radius: 20px;
      padding: 30px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    h1 {
      color: #333;
      text-align: center;
      margin-bottom: 10px;
      font-size: 28px;
    }
    .time-display {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 20px;
      border-radius: 15px;
      text-align: center;
      margin: 20px 0;
      font-size: 24px;
      font-weight: bold;
    }
    .feed-buttons {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 15px;
      margin: 20px 0;
    }
    .feed-btn {
      background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
      color: white;
      border: none;
      padding: 20px;
      border-radius: 15px;
      font-size: 18px;
      font-weight: bold;
      cursor: pointer;
      transition: transform 0.2s;
      box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .feed-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .feed-btn:active {
      transform: translateY(0);
    }
    .section {
      background: #f8f9fa;
      padding: 20px;
      border-radius: 15px;
      margin: 15px 0;
    }
    .section h2 {
      color: #667eea;
      font-size: 18px;
      margin-bottom: 15px;
    }
    .schedule-item {
      background: white;
      padding: 12px;
      border-radius: 10px;
      margin: 8px 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .btn {
      background: #667eea;
      color: white;
      border: none;
      padding: 12px 24px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 16px;
      width: 100%;
      margin: 10px 0;
      transition: background 0.3s;
    }
    .btn:hover {
      background: #5568d3;
    }
    .status {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: bold;
    }
    .status-done { background: #d4edda; color: #155724; }
    .status-pending { background: #fff3cd; color: #856404; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🐟 自動餌やりシステム</h1>
    
    <div class="time-display" id="currentTime">
      )rawliteral";
  
  char timeStr[50];
  sprintf(timeStr, "%04d/%02d/%02d %02d:%02d:%02d",
          now.year(), now.month(), now.day(),
          now.hour(), now.minute(), now.second());
  html += String(timeStr);
  
  html += R"rawliteral(
    </div>
    
    <div class="feed-buttons">
      <button class="feed-btn" onclick="feed(1)">🐟 サーボ1<br>餌やり</button>
      <button class="feed-btn" onclick="feed(2)">🐟 サーボ2<br>餌やり</button>
    </div>
    
    <div class="section">
      <h2>📅 本日のスケジュール</h2>
      <div class="schedule-item">
        <span>9時 - サーボ1</span>
        <span class="status )rawliteral";
  
  html += fed_9am ? "status-done\">完了" : "status-pending\">待機中";
  html += R"rawliteral(</span>
      </div>
      <div class="schedule-item">
        <span>12時 - サーボ1</span>
        <span class="status )rawliteral";
  
  html += fed_12pm_servo1 ? "status-done\">完了" : "status-pending\">待機中";
  html += R"rawliteral(</span>
      </div>
      <div class="schedule-item">
        <span>12時 - サーボ2</span>
        <span class="status )rawliteral";
  
  html += fed_12pm_servo2 ? "status-done\">完了" : "status-pending\">待機中";
  html += R"rawliteral(</span>
      </div>
      <div class="schedule-item">
        <span>16時 - サーボ1</span>
        <span class="status )rawliteral";
  
  html += fed_4pm ? "status-done\">完了" : "status-pending\">待機中";
  html += R"rawliteral(</span>
      </div>
    </div>
    
    <button class="btn" onclick="location.href='/logs'">📜 餌やり記録</button>
    <button class="btn" onclick="location.href='/config'">⚙️ 設定変更</button>
  </div>
  
  <script>
    function feed(servo) {
      if(confirm('サーボ' + servo + 'で餌やりを実行しますか？')) {
        fetch('/feed?servo=' + servo)
          .then(response => response.text())
          .then(data => {
            alert(data);
            location.reload();
          });
      }
    }
    
    // 1秒ごとに時刻更新
    setInterval(() => {
      fetch('/status')
        .then(response => response.json())
        .then(data => {
          document.getElementById('currentTime').textContent = data.time;
        });
    }, 1000);
  </script>
</body>
</html>
  )rawliteral";
  
  server.send(200, "text/html", html);
}

// 手動餌やり
void handleManualFeed() {
  String servoNum = server.arg("servo");
  
  if (servoNum == "1") {
    feedServo(servo1, config.servo1_rotations, "サーボ1", "手動");
    server.send(200, "text/plain", "サーボ1で餌やり完了！");
  } else if (servoNum == "2") {
    feedServo(servo2, config.servo2_rotations, "サーボ2", "手動");
    server.send(200, "text/plain", "サーボ2で餌やり完了！");
  } else {
    server.send(400, "text/plain", "エラー: 無効なサーボ番号");
  }
}

// 餌やり記録ページ
void handleLogs() {
  String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>📜 餌やり記録</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 20px;
    }
    .container {
      max-width: 600px;
      margin: 0 auto;
      background: white;
      border-radius: 20px;
      padding: 30px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    h1 {
      color: #333;
      text-align: center;
      margin-bottom: 20px;
    }
    .log-item {
      background: #f8f9fa;
      padding: 15px;
      border-radius: 10px;
      margin: 10px 0;
      border-left: 4px solid #667eea;
    }
    .log-time {
      font-weight: bold;
      color: #667eea;
      margin-bottom: 5px;
    }
    .log-detail {
      color: #666;
      font-size: 14px;
    }
    .btn {
      background: #667eea;
      color: white;
      border: none;
      padding: 12px 24px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 16px;
      width: 100%;
      margin: 20px 0 0 0;
    }
    .empty {
      text-align: center;
      color: #999;
      padding: 40px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>📜 餌やり記録（最新10件）</h1>
  )rawliteral";
  
  bool hasLogs = false;
  for (int i = 9; i >= 0; i--) {
    int idx = (logIndex - 1 - i + 10) % 10;
    if (logs[idx].timestamp != "") {
      hasLogs = true;
      html += "<div class='log-item'>";
      html += "<div class='log-time'>" + logs[idx].timestamp + "</div>";
      html += "<div class='log-detail'>" + logs[idx].servo + " - " + logs[idx].type + "</div>";
      html += "</div>";
    }
  }
  
  if (!hasLogs) {
    html += "<div class='empty'>まだ餌やり記録がありません</div>";
  }
  
  html += R"rawliteral(
    <button class="btn" onclick="location.href='/'">🏠 ホームに戻る</button>
  </div>
</body>
</html>
  )rawliteral";
  
  server.send(200, "text/html", html);
}

// 設定ページ
void handleConfig() {
  if (server.method() == HTTP_POST) {
    // 設定を保存
    config.servo1_rotations = server.arg("s1_rot").toInt();
    config.servo2_rotations = server.arg("s2_rot").toInt();
    config.feed_hour_1 = server.arg("hour1").toInt();
    config.feed_hour_2 = server.arg("hour2").toInt();
    config.feed_hour_3 = server.arg("hour3").toInt();
    
    preferences.putInt("s1_rot", config.servo1_rotations);
    preferences.putInt("s2_rot", config.servo2_rotations);
    preferences.putInt("hour1", config.feed_hour_1);
    preferences.putInt("hour2", config.feed_hour_2);
    preferences.putInt("hour3", config.feed_hour_3);
    
    server.send(200, "text/html", 
      "<html><body><h1>設定を保存しました</h1>"
      "<script>setTimeout(() => location.href='/', 2000);</script></body></html>");
    return;
  }
  
  String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>⚙️ 設定</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 20px;
    }
    .container {
      max-width: 600px;
      margin: 0 auto;
      background: white;
      border-radius: 20px;
      padding: 30px;
    }
    h1 { color: #333; text-align: center; margin-bottom: 20px; }
    .form-group {
      margin: 20px 0;
      padding: 15px;
      background: #f8f9fa;
      border-radius: 10px;
    }
    label {
      display: block;
      margin-bottom: 8px;
      color: #667eea;
      font-weight: bold;
    }
    input {
      width: 100%;
      padding: 12px;
      border: 2px solid #ddd;
      border-radius: 8px;
      font-size: 16px;
    }
    .btn {
      background: #667eea;
      color: white;
      border: none;
      padding: 15px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 16px;
      width: 100%;
      margin: 10px 0;
    }
    .btn-secondary {
      background: #6c757d;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>⚙️ 設定変更</h1>
    <form method="POST">
      <div class="form-group">
        <label>サーボ1 回転回数</label>
        <input type="number" name="s1_rot" value=")rawliteral" + String(config.servo1_rotations) + R"rawliteral(" min="1" max="10">
      </div>
      <div class="form-group">
        <label>サーボ2 回転回数</label>
        <input type="number" name="s2_rot" value=")rawliteral" + String(config.servo2_rotations) + R"rawliteral(" min="1" max="10">
      </div>
      <div class="form-group">
        <label>餌やり時刻1（サーボ1）</label>
        <input type="number" name="hour1" value=")rawliteral" + String(config.feed_hour_1) + R"rawliteral(" min="0" max="23">
      </div>
      <div class="form-group">
        <label>餌やり時刻2（サーボ1+2）</label>
        <input type="number" name="hour2" value=")rawliteral" + String(config.feed_hour_2) + R"rawliteral(" min="0" max="23">
      </div>
      <div class="form-group">
        <label>餌やり時刻3（サーボ1）</label>
        <input type="number" name="hour3" value=")rawliteral" + String(config.feed_hour_3) + R"rawliteral(" min="0" max="23">
      </div>
      <button type="submit" class="btn">💾 保存</button>
      <button type="button" class="btn btn-secondary" onclick="location.href='/'">キャンセル</button>
    </form>
  </div>
</body>
</html>
  )rawliteral";
  
  server.send(200, "text/html", html);
}

// 時刻設定
void handleSetTime() {
  if (server.method() == HTTP_POST) {
    int year = server.arg("year").toInt();
    int month = server.arg("month").toInt();
    int day = server.arg("day").toInt();
    int hour = server.arg("hour").toInt();
    int minute = server.arg("minute").toInt();
    
    rtc.adjust(DateTime(year, month, day, hour, minute, 0));
    
    server.send(200, "text/html",
      "<html><body><h1>時刻を設定しました</h1>"
      "<script>setTimeout(() => location.href='/', 2000);</script></body></html>");
    return;
  }
  
  DateTime now = rtc.now();
  
  String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🕐 時刻設定</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 20px;
    }
    .container {
      max-width: 500px;
      margin: 0 auto;
      background: white;
      border-radius: 20px;
      padding: 30px;
    }
    h1 { text-align: center; margin-bottom: 20px; }
    .form-group {
      margin: 15px 0;
    }
    label {
      display: block;
      margin-bottom: 5px;
      font-weight: bold;
      color: #667eea;
    }
    input {
      width: 100%;
      padding: 10px;
      border: 2px solid #ddd;
      border-radius: 8px;
      font-size: 16px;
    }
    .btn {
      background: #667eea;
      color: white;
      border: none;
      padding: 15px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 16px;
      width: 100%;
      margin: 10px 0;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🕐 時刻設定</h1>
    <form method="POST">
      <div class="form-group">
        <label>年</label>
        <input type="number" name="year" value=")rawliteral" + String(now.year()) + R"rawliteral(" required>
      </div>
      <div class="form-group">
        <label>月</label>
        <input type="number" name="month" value=")rawliteral" + String(now.month()) + R"rawliteral(" min="1" max="12" required>
      </div>
      <div class="form-group">
        <label>日</label>
        <input type="number" name="day" value=")rawliteral" + String(now.day()) + R"rawliteral(" min="1" max="31" required>
      </div>
      <div class="form-group">
        <label>時</label>
        <input type="number" name="hour" value=")rawliteral" + String(now.hour()) + R"rawliteral(" min="0" max="23" required>
      </div>
      <div class="form-group">
        <label>分</label>
        <input type="number" name="minute" value=")rawliteral" + String(now.minute()) + R"rawliteral(" min="0" max="59" required>
      </div>
      <button type="submit" class="btn">💾 時刻を設定</button>
      <button type="button" class="btn" style="background:#6c757d" onclick="location.href='/'">戻る</button>
    </form>
  </div>
</body>
</html>
  )rawliteral";
  
  server.send(200, "text/html", html);
}

// ステータスAPI（時刻更新用）
void handleStatus() {
  DateTime now = rtc.now();
  char timeStr[20];
  sprintf(timeStr, "%04d/%02d/%02d %02d:%02d:%02d",
          now.year(), now.month(), now.day(),
          now.hour(), now.minute(), now.second());
  
  String json = "{\"time\":\"" + String(timeStr) + "\"}";
  server.send(200, "application/json", json);
}
