### Diagrama de Classes do Banco

``` mermaid
classDiagram
    class Conta {
        - String id
        # double saldo
        - Cliente cliente
        + sacar(double valor)
        + depositar(double valor)
    }
    class Cliente {
        - String id
        - String nome
        - List<Conta> contas
    }
    class PessoaFisica {
        - String cpf
    }
    class PessoaJuridica {
        - String cnpj
    }
    class ContaCorrente {
        - double limite
        + sacar(double valor)
    }
    class ContaPoupanca {
        + sacar(double valor)
    }
    Conta *-- Cliente
    Conta <|-- ContaCorrente
    Conta <|-- ContaPoupanca
    Cliente <|-- PessoaFisica
    Cliente <|-- PessoaJuridica
```

### Diagrama de Seqüência de Autorização

``` mermaid
sequenceDiagram
  autonumber
  actor User
  User->>Auth Service: request with token
  Auth Service->>Auth Service: decodes the token and extracts claims
  Auth Service->>Auth Service: verifies permissions
  critical allowed
    Auth Service->>Secured Resource: authorizes the request
    Secured Resource->>User: returns the response
  option denied
    Auth Service-->>User: unauthorized message
  end  
```
