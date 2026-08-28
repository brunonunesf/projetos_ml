# Regressão linear — geração de energia

Estudo de regressão linear múltipla para estimar a geração elétrica de uma
usina a partir de variáveis ambientais.

## Dados

O arquivo `usina.csv` possui 9.568 observações e as seguintes colunas:

- `AT`: temperatura ambiente;
- `V`: vácuo de exaustão;
- `AP`: pressão ambiente;
- `RH`: umidade relativa;
- `PE`: geração elétrica, variável que o modelo busca prever.

## Conceitos praticados

- regressão por mínimos quadrados ordinários (OLS);
- coeficiente de determinação (R²);
- fator de inflação da variância (VIF);
- análise visual de resíduos.

## Execução

A partir da raiz do repositório:

```bash
python regressao-linear/main.py
```

Ao executar, o programa exibe as métricas no terminal e abre dois gráficos
interativos no navegador.
