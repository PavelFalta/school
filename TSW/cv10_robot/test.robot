*** Settings ***
Library    BuiltIn
Library    Collections
Library    String
Library    Dialogs
Library    JSONLibrary
Library    SeleniumLibrary

*** Variables ***
${TEXT}    Hello Robot Framework
${LIST}    apple    banana   cherry
${URL}     http://localhost:5000

*** Test Cases ***
My First Test
    Should Contain      ${TEXT}     Robot

Simple String Check
    Should Contain    Hello Robot Framework    Robot

Check Item In List
    ${mylist}=      Create List     apple       banana      orange
    List Should Contain Value       ${mylist}     banana

Check That Five Is Greater Than Three
    Should Be True    ${5} > ${3}

Check That Sum Is Correct
    ${result}=    Evaluate    2 + 2
    Should Be Equal As Numbers    ${result}    4

Text Contains Keyword
    Should Contain    This is a test string    test

Check Exact Match
    Should Be Equal    Hello    Hello

Text Does Not Contain
    Should Not Contain    Hello world    goodbye

Check List Length
    ${mylist}=    Create List    apple    banana    orange
    Length Should Be    ${mylist}    3

Check Item In List
    ${mylist}=    Create List    apple    banana    orange
    Should Contain    ${mylist}    banana

Convert To Int And Multiply
    ${val}=    Convert To Integer    7
    ${result}=    Evaluate    ${val} * 3
    Should Be Equal As Numbers    ${result}    21

Boolean Test
    ${val}=    Set Variable    ${True}
    Should Be True    ${val}

Check Concatenation
    ${first}=    Set Variable    Hello
    ${second}=    Set Variable    World
    ${joined}=    Catenate    SEPARATOR=    ${first}    ${second}
    Should Be Equal    ${joined}    HelloWorld

Should Be Equal Example
    ${a}=   Set Variable    42
    ${b}=   Evaluate    6 * 7
    Should Be Equal As Numbers     ${a}     ${b}

Replace Substring Example
    ${text}=    Set Variable    Hello student
    ${new}=     Replace String      ${text}     student     teacher
    Should Be Equal     ${new}      Hello teacher

List Contains Value Example
    ${fruits}=      Create List     apple   banana  cherry
    List Should Contain Value   ${fruits}   banana

Get Value From JSON Example
    ${json}=    Convert String To Json      {"name": "Alice", "age": 30}
    ${age_list}=    Get Value From Json     ${json}     $.age
    ${age}=     Get From List   ${age_list}     0
    Should Be Equal As Numbers  ${age}   30

Add New Task
    Open Browser    ${URL}      chrome
    Input Text      name=task   Napsat test
    Click Button    xpath=//input[@type="submit"]
    Page Should Contain     Odesláno
    Close Browser