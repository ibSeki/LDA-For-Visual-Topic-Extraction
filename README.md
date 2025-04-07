```markdown
# 📚 LDADE – LDA com Otimização Diferencial

Este projeto implementa uma variação do algoritmo LDA (Latent Dirichlet Allocation) utilizando estratégias de **Otimização Diferencial (Differential Evolution)** para extração de tópicos mais eficientes, especialmente em contextos visuais e de grandes volumes de dados.

> 🔍 Baseado no repositório original de [Amritanshu Agrawal](https://github.com/amritbhanu/LDADE-package/blob/master/LDADE/Tests/LDADE_test.py).

## 🚀 Funcionalidades

- Aplicação de LDA em documentos
- Otimização de hiperparâmetros via Evolução Diferencial
- Scripts de teste prontos para uso
- Estrutura modular com pacotes reutilizáveis

## 🛠️ Tecnologias Utilizadas

- Python 2.7 ou compatível com Python 3*
- NumPy
- Scikit-learn

\* *É recomendável atualizar para Python 3 e ajustar a sintaxe, se necessário.*

## 📁 Estrutura do Projeto

```
LDADE-package-master/
├── LDADE/
│   ├── DE.py             # Implementação da Otimização Diferencial
│   ├── LDADE.py          # Execução principal do algoritmo LDADE
│   ├── LDADE_test.py     # Testes e exemplo de uso
│   └── __init__.py
├── setup.py
├── LICENSE
└── README.md
```

## 📦 Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/ibSeki/LDADE-package.git
   cd LDADE-package-master
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute o script de teste:
   ```bash
   python LDADE/LDADE_test.py
   ```

## 📩 Contato

Desenvolvido e adaptado por **Ian de Barros Seki**  
📧 ian.dbseki@gmail.com  
🔗 [GitHub - ibSeki](https://github.com/ibSeki)
```
