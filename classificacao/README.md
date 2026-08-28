# Classificação de projetos

Modelo de classificação que prevê se um projeto será finalizado usando o preço
e a quantidade de horas esperadas como variáveis de entrada.

## Conceitos praticados

- classificação binária com `LinearSVC`;
- separação estratificada entre treino e teste;
- avaliação do modelo por acurácia.

## Execução

A partir da raiz do repositório:

```bash
python classificacao/main.py
```

Os dados são carregados de uma fonte pública, portanto a execução requer acesso
à internet.
