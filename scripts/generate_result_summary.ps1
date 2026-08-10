param(
    [string]$ResultPath = "result",
    [string]$SnapshotDate = (Get-Date -Format "yyyy-MM-dd")
)

$inv = [System.Globalization.CultureInfo]::InvariantCulture
$taskOrder = @{ arc_challenge = 0; arc_easy = 1; gsm8k = 2; humanevalplus = 3; mbppplus = 4; medqa = 5 }
$modelOrder = @{ "Qwen/Qwen3-8B" = 0; "Qwen/Qwen3-14B" = 1 }
$methodOrder = @{ baseline = 0; latent_mas = 1; text_mas = 2 }
$alignOrder = @{ identical = 0; kernel = 1; linear = 2; soft = 3 }
$topologyOrder = @{ hierarchical = 0; sequential = 1 }
$taskNames = @{
    arc_challenge = "ARC Challenge"
    arc_easy = "ARC Easy"
    gsm8k = "GSM8K"
    humanevalplus = "HumanEval+"
    mbppplus = "MBPP+"
    medqa = "MedQA"
}

$rows = foreach ($file in (Get-ChildItem -LiteralPath $ResultPath -Recurse -File -Filter summary.json)) {
    $summary = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
    [pscustomobject]@{
        task = [string]$summary.average.run.task
        model = [string]$summary.average.run.model
        method = [string]$summary.average.run.method
        align = [string]$summary.average.run.align_method
        topology = [string]$summary.average.run.prompt
        processed = [double]$summary.average.results.processed
        timing = [double]$summary.average.timing.total_seconds
        accuracy = [double]$summary.average.results.accuracy
        tokens = [double]$summary.average.results.tokens.text_output.total
        repetitions = [int]$summary.aggregation.repetitions
        seeds = $summary.aggregation.seeds -join ", "
    }
}

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# 实验结果汇总对比")
$lines.Add("")
$lines.Add("本文档汇总当前 ``result`` 目录所有子文件夹中的 ``summary.json``。共收录 **$($rows.Count)** 组结果；所有 split 均为 ``test``。")
$lines.Add("")
$lines.Add("指标定义：")
$lines.Add("")
$lines.Add("- ``Timing (total)``：``average.timing.total_seconds``，单位为秒，越低越好。")
$lines.Add("- ``Accuracy``：``average.results.accuracy``，越高越好；括号内为原始小数值。")
$lines.Add("- ``Text output tokens``：``average.results.tokens.text_output.total``，越低表示文本输出开销越少。")
$lines.Add("- 表格中的数值均来自 ``average``；多次运行的结果可能带小数。")
$lines.Add("- 粗体表示同一任务、同一模型下的单项最优值。")

foreach ($taskGroup in ($rows | Group-Object task | Sort-Object { $taskOrder[$_.Name] })) {
    $taskTitle = if ($taskNames.ContainsKey($taskGroup.Name)) { $taskNames[$taskGroup.Name] } else { $taskGroup.Name }
    $lines.Add("")
    $lines.Add("## $taskTitle")

    foreach ($modelGroup in ($taskGroup.Group | Group-Object model | Sort-Object { $modelOrder[$_.Name] })) {
        $group = @($modelGroup.Group)
        $samples = @($group.processed | Sort-Object -Unique)
        $repetitions = @($group.repetitions | Sort-Object -Unique)
        $seeds = @($group.seeds | Sort-Object -Unique)
        $sampleText = ($samples | ForEach-Object { $_.ToString("0.####", $inv) }) -join "/"
        $repetitionText = ($repetitions | ForEach-Object { $_.ToString($inv) }) -join "/"

        $lines.Add("")
        $lines.Add("### $($modelGroup.Name)")
        $lines.Add("")
        $lines.Add("样本数：$sampleText；汇总组数：$($group.Count)；重复次数：$repetitionText；seeds：$($seeds -join ' / ')。")
        $lines.Add("")
        $lines.Add("| Method | Align method | Topology | Timing (total, s) ↓ | Accuracy ↑ | Text output tokens ↓ |")
        $lines.Add("|---|---|---|---:|---:|---:|")

        $minTiming = ($group | Measure-Object timing -Minimum).Minimum
        $maxAccuracy = ($group | Measure-Object accuracy -Maximum).Maximum
        $minTokens = ($group | Measure-Object tokens -Minimum).Minimum
        $sorted = $group | Sort-Object `
            @{ Expression = { $methodOrder[$_.method] } }, `
            @{ Expression = { $alignOrder[$_.align] } }, `
            @{ Expression = { $topologyOrder[$_.topology] } }

        foreach ($row in $sorted) {
            $method = switch ($row.method) {
                "baseline" { "Baseline" }
                "latent_mas" { "Latent MAS" }
                "text_mas" { "Text MAS" }
                default { $row.method }
            }
            $timing = $row.timing.ToString("F4", $inv)
            $accuracy = ($row.accuracy * 100).ToString("F4", $inv) + "% (" + $row.accuracy.ToString("F6", $inv) + ")"
            $tokens = if ([math]::Abs($row.tokens - [math]::Round($row.tokens)) -lt 0.0000001) {
                $row.tokens.ToString("N0", $inv)
            } else {
                $row.tokens.ToString("N2", $inv)
            }

            if ($row.timing -eq $minTiming) { $timing = "**$timing**" }
            if ($row.accuracy -eq $maxAccuracy) { $accuracy = "**$accuracy**" }
            if ($row.tokens -eq $minTokens) { $tokens = "**$tokens**" }
            $lines.Add("| $method | $($row.align) | $($row.topology) | $timing | $accuracy | $tokens |")
        }
    }
}

$lines.Add("")
$lines.Add("数据快照日期：$SnapshotDate。仅收录实际存在且可解析的 ``summary.json``。")
$lines -join "`n"
