#!/bin/bash

# Ollamaサーバーをバックグラウンドで起動
ollama serve &
pid=$!

# サーバーが立ち上がるまで少し待機
sleep 5

echo "🔴 モデルのダウンロードを開始します..."

echo "🔴 トラブルシューティングのため、軽量なgemma:2bモデルのみをダウンロードします..."
ollama pull gemma:2b

echo "🟢 すべてのモデルの準備が完了しました！"

# プロセスが終了しないように待機
wait $pid
