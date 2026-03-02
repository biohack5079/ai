#!/bin/bash

# Spaceの設定画面(Variables)での入力ミスを強制的に修正
export OLLAMA_HOST=0.0.0.0:7860
# export OLLAMA_ORIGINS='*'

# Ollamaサーバーをバックグラウンドで起動
ollama serve &
pid=$!

# サーバーが立ち上がるまで少し待機
sleep 5

echo "🔴 モデルのダウンロードを開始します..."

# モデルのダウンロードを並列で実行して高速化します
(
    echo "--- Pulling gemma:7b ---"
    ollama pull gemma:7b
) &

(
    echo "--- Pulling qwen:14b ---"
    ollama pull qwen:14b
) &

# バックグラウンドで実行したダウンロード処理がすべて完了するのを待つ
wait

echo "--- Creating alias for gpt-oss:20b ---"
ollama cp qwen:14b gpt-oss:20b

echo "🟢 すべてのモデルの準備が完了しました！"

# プロセスが終了しないように待機
wait $pid
