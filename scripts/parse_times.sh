#!/usr/bin/env bash

# exp0: fast/ (completo: toy+mnist+sicap+pathmnist con config actual)
# exp1, exp2, exp5: logs/ root (MLP toy/mnist/sicap, no han cambiado)
# exp3, exp4: fast/ (UNet mnist y pathmnist reentrenados con arquitectura actual)
# log_4_vae*_eval.txt excluidos: son runs eval-only (~20 min), sobrescriben los tiempos de training reales
_log4_files=$(ls logs/fast/log_4_*.txt 2>/dev/null | grep -v "log_4_vae.*_eval\.txt")
logs="${*:-logs/fast/log_0.txt logs/log_1.txt logs/log_2.txt logs/fast/log_3_*.txt $_log4_files logs/log_5.txt}"

for log in $logs; do
    [[ -f "$log" ]] || continue
    echo "── $(basename "$log") ──"

    grep -E '\[time\]|\[info\].*\[([a-zA-Z0-9_]+)\].*seed=' "$log" \
    | while IFS= read -r line; do
        if [[ "$line" =~ \[time\][[:space:]]+([^[:space:]]+)[[:space:]]+\-\>[[:space:]]+([0-9:]+) ]]; then
            printf "  %-45s %s\n" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
        fi
    done

    awk '
    /\[time\]/ {
        match($0, /\[time\][[:space:]]+([^[:space:]]+)[[:space:]]+->.*([0-9]{2}):([0-9]{2}):([0-9]{2})/, a)
        split(a[1], p, "/")
        s = a[2]*3600 + a[3]*60 + a[4]
        model[p[1]] += s; model_n[p[1]]++
        dataset[p[2]] += s; dataset_n[p[2]]++
        total += s; total_n++
    }
    function fmt(s) { return sprintf("%02d:%02d:%02d", s/3600, (s%3600)/60, s%60) }
    function row(name, n, s) { printf "  %-28s %4d    %s\n", name, n, fmt(s) }
    END {
        printf "\n  %-28s %4s    %s\n", "NOMBRE", "RUNS", "TOTAL"
        print  "  " sprintf("%-44s", "─────────────────────────────────────────────")
        print  "  Modelo"
        for (m in model)   row("    " m, model_n[m], model[m])
        print  "  Dataset"
        for (d in dataset) row("    " d, dataset_n[d], dataset[d])
        print  "  " sprintf("%-44s", "─────────────────────────────────────────────")
        row("  TOTAL", total_n, total)
    }' "$log"

    echo ""
done
