Output "docs/images/demo.mp4"

Set Shell bash
Set Theme Dracula
Set Width 1400
Set Height 900
Set FontSize 15
Set Padding 20
Set TypingSpeed 25ms
Set PlaybackSpeed 2

# Hidden setup — activate venv so imgeda is on PATH
Hide
Type "cd /Users/ranman/dev/caylent/customers/gsm-outdoors"
Enter
Type "source .venv/bin/activate"
Enter
Type "export FORCE_COLOR=1"
Enter
Type "rm -f /tmp/imgeda_demo.jsonl /tmp/imgeda_demo_report.html && rm -rf /tmp/imgeda_plots/"
Enter
Type "clear"
Enter
Sleep 0.5s
Show

# Show version
Type "imgeda --version"
Enter
Sleep 2s

# Scan the Oxford-IIIT Pet Dataset (3,680 images)
Type "imgeda scan ./data/oxford-pet -o /tmp/imgeda_demo.jsonl"
Enter
Sleep 15s

# Show dataset summary
Type "imgeda info -m /tmp/imgeda_demo.jsonl"
Enter
Sleep 3s

# Check for quality issues
Type "imgeda check all -m /tmp/imgeda_demo.jsonl"
Enter
Sleep 5s

# Generate all plots
Type "imgeda plot all -m /tmp/imgeda_demo.jsonl -o /tmp/imgeda_plots"
Enter
Sleep 10s

# Generate HTML report
Type "imgeda report -m /tmp/imgeda_demo.jsonl -o /tmp/imgeda_demo_report.html"
Enter
Sleep 8s

Sleep 2s
