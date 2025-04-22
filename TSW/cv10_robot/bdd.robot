*** Settings ***
Library    SeleniumLibrary

*** Variables ***
${URL}    http://localhost:5000

*** Test Cases ***
Adding A Task Should Appear In List
    Given Browser Is Open To Todo Page
    When I Add Task With Text    Udělat kávu
    Then I Should See Task In List    Udělat kávu

*** Keywords ***
Browser Is Open To Todo Page
    Open Browser    ${URL}    chrome

I Add Task With Text
    [Arguments]    ${text}
    Input Text    name=task    ${text}
    Click Button    xpath=//input[@type="submit"]

I Should See Task In List
    [Arguments]    ${text}
    Page Should Contain    ${text}
    Close Browser