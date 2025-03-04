Feature: Bank Account

    Scenario: Vklad peněz na účet
        Given nový bankovní účet
        When vložím 200 Kč
        Then zůstatek je 200 Kč
    
    Scenario: Výběr peněz úspěch
        Given bankovní účet s 200 Kč
        When vyberu 100 Kč
        Then zůstatek je 100 Kč

    Scenario: Výběr peněz neúspěch
        Given nový bankovní účet
        When vyberu 500 Kč
        Then zůstatek je 0 Kč