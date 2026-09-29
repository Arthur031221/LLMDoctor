# Render from the repository root after `uv sync`:  vhs demo/demo.tape
# Needs Ollama running with qwen3:1.7b pulled.

Output demo/demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 1400
Set Height 860
Set Padding 16
Set TypingSpeed 35ms
Set Theme "Catppuccin Mocha"

Hide
Type 'export PATH="$PWD/.venv/bin:$PATH" COLUMNS=150 && clear'
Enter
Show

Type "llm-doctor scan"
Sleep 400ms
Enter
Wait@90s
Sleep 4s

Type "clear && llm-doctor endpoint http://localhost:11434 --model qwen3:1.7b --only context,tools,think"
Sleep 400ms
Enter
Wait@120s
Sleep 6s
