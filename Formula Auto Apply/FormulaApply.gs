const monthsMap = 
[
  {month: 'Jan', val: '01'},
  {month: 'Feb', val: '02'},
  {month: 'Mar', val: '03'},
  {month: 'Apr', val: '04'},
  {month: 'May', val: '05'},
  {month: 'Jun', val: '06'},
  {month: 'Jul', val: '07'},
  {month: 'Aug', val: '08'},
  {month: 'Sept', val: '09'},
  {month: 'Oct', val: '10'},
  {month: 'Nov', val: '11'},
  {month: 'Dec', val: '12'}
];
let weeklyCol = 'G';
let nameCol = 'A';
let blueCode = '#a4c2f4';
let greenCode = '#b6d7a8';
let yellowCode = '#ffe599';
let grayCode = '#666666';
let end = '#c9daf8';
let colorArr = ['blue', 'green', 'yellow'];
let colorCodeArr = [blueCode, greenCode, yellowCode];
let sheetNameArr = ["weekly contribution", "benevolent", "Mission", "Miscellaneous"];

/**
 * get the formula based on subsheet 
 */
function getFormulaBySheet(sheetName, newDate, rowNum, color) {
  if (sheetName == "weekly contribution") {
    return getWeeklyContri(newDate, rowNum, color);
  } else if (sheetName == "benevolent") {
    return getBenev(newDate, rowNum, color);
  } else if (sheetName == "Mission") {
    return getMission(newDate, rowNum, color);
  } else if (sheetName == "Miscellaneous"){
    return getMiscellaneous(newDate, rowNum, color);
  }
  return null;
}

/**
 * get weekly contri bution formula based on section 
 */
function getWeeklyContri(newDate, rowNum, color) {
  let url;
  if (color == 'blue') {
    url = "https://docs.google.com/spreadsheets/d/1J0-BzNBSFIZMava_xPekOV54wny-e6gkkZo0tINnPno/edit#gid=1333242328";
  } else if (color == 'green') {
    url = "https://docs.google.com/spreadsheets/d/1V56Qmjfr0eol-QTol3C0Jv8O9V67mhfn3vVrJbPfoxA/edit#gid=0";
  } else {
    url = "https://docs.google.com/spreadsheets/d/1AXICpVaAggJvMytYy9MRBIK2rBRSkw9TYGGPKRzlEN0/edit#gid=0";
  }
  
  let currFormula = "=if(QUERY(IMPORTRANGE(\"" + url + "\",\"'" + newDate + "'!D11:J\"),\"Select Col7 where Col2 contains '\"&$B" + rowNum + "&\"' and Col1 contains '\"&$A" + rowNum + "&\"'\")=0,\"\",QUERY(IMPORTRANGE(\"" + url + "\",\"'" + newDate + "'!D11:J\"),\"Select Col7 where Col2 contains '\"&$B" + rowNum + "&\"' and Col1 contains '\"&$A" + rowNum + "&\"'\"))";

  return currFormula;
}

/**
 * get benev formula based on section
 */
function getBenev(newDate, rowNum, color) {
  let url;
  if (color == 'blue') {
    url = "https://docs.google.com/spreadsheets/d/1J0-BzNBSFIZMava_xPekOV54wny-e6gkkZo0tINnPno/edit#gid=1333242328";
  } else if (color == 'green') {
    url = "https://docs.google.com/spreadsheets/d/1V56Qmjfr0eol-QTol3C0Jv8O9V67mhfn3vVrJbPfoxA/edit#gid=0";
  } else {
    url = "https://docs.google.com/spreadsheets/d/1AXICpVaAggJvMytYy9MRBIK2rBRSkw9TYGGPKRzlEN0/edit#gid=0";
  }

  let currFormula = "=if(QUERY(IMPORTRANGE(\"" + url + "\",\"'" + newDate + "'!D11:AA\"),\"Select Col24 where Col2 contains '\"&$B" + rowNum + "&\"' and Col1 contains '\"&$A" + rowNum + "&\"'\")=0,\"\",QUERY(IMPORTRANGE(\"" + url + "\",\"'" + newDate + "'!D11:AA\"),\"Select Col24 where Col2 contains '\"&$B" + rowNum + "&\"' and Col1 contains '\"&$A" + rowNum + "&\"'\"))";

  return currFormula;
}

/**
 * get mission formula based on section 
 */
function getMission(newDate, rowNum, color) {
  let url;
  if (color == 'blue') {
    url = "https://docs.google.com/spreadsheets/d/1J0-BzNBSFIZMava_xPekOV54wny-e6gkkZo0tINnPno/edit#gid=1333242328";
  } else if (color == 'green') {
    url = "https://docs.google.com/spreadsheets/d/1V56Qmjfr0eol-QTol3C0Jv8O9V67mhfn3vVrJbPfoxA/edit#gid=0";
  } else {
    url = "https://docs.google.com/spreadsheets/d/1AXICpVaAggJvMytYy9MRBIK2rBRSkw9TYGGPKRzlEN0/edit#gid=0";
  }

  let currFormula = "=if(QUERY(IMPORTRANGE(\"" + url + "\",\"'" + newDate + "'!D11:O\"),\"Select Col12 where Col2 contains '\"&$B" + rowNum + "&\"' and Col1 contains '\"&$A" + rowNum + "&\"'\")=0,\"\",QUERY(IMPORTRANGE(\"" + url + "\",\"'" + newDate + "'!D11:O\"),\"Select Col12 where Col2 contains '\"&$B" + rowNum + "&\"' and Col1 contains '\"&$A" + rowNum + "&\"'\"))";

  return currFormula;
}

/**
 * get miscellaneous formula based on section 
 */
function getMiscellaneous(newDate, rowNum, color) {
  let url;
  if (color == 'blue') {
    url = "https://docs.google.com/spreadsheets/d/1J0-BzNBSFIZMava_xPekOV54wny-e6gkkZo0tINnPno/edit#gid=1333242328";
  } else if (color == 'green') {
    url = "https://docs.google.com/spreadsheets/d/1V56Qmjfr0eol-QTol3C0Jv8O9V67mhfn3vVrJbPfoxA/edit#gid=0";
  } else {
    url = "https://docs.google.com/spreadsheets/d/1AXICpVaAggJvMytYy9MRBIK2rBRSkw9TYGGPKRzlEN0/edit#gid=0";
  }

  let currFormula = "=if(QUERY(IMPORTRANGE(\"" + url + "\",\"'" + newDate + "'!D11:AF\"),\"Select Col29 where Col2 contains '\"&$B" + rowNum + "&\"' and Col1 contains '\"&$A" + rowNum + "&\"'\")=0,\"\",QUERY(IMPORTRANGE(\"" + url + "\",\"'" + newDate + "'!D11:AF\"),\"Select Col29 where Col2 contains '\"&$B" + rowNum + "&\"' and Col1 contains '\"&$A" + rowNum + "&\"'\"))";

  return currFormula;
}

/*
 * This is the helper function for applying formula to
 * different section.
 * parameters: 
 *   rowNum: row number
 *   color: color name, note: you can only pass blue, green, yellow
 */
function applyFormula(sheet, sheetName, date) {
  let color = colorArr[0];
  let colorCode = colorCodeArr[0];
  let dataCol = weeklyCol;
  let currColor = 0;

  rowNum = 2;

  while(true) {
    cell = sheet.getRange(dataCol+rowNum);
    if (sheet.getRange('A'+rowNum).getBackground() == end) {
      break;
    }
    // apply formula
    weeklyFormula = getFormulaBySheet(sheetName, date, rowNum, color);
    if (sheet.getRange(nameCol+rowNum).getBackground() != colorCode &&
        sheet.getRange(nameCol+rowNum).getBackground() != grayCode) {
        currColor++;
        color = colorArr[currColor];
        colorCode = colorCodeArr[currColor];
        weeklyFormula = getFormulaBySheet(sheetName, date, rowNum, color);
        cell.setFormula(weeklyFormula);
    } else if (sheet.getRange(nameCol+rowNum).getBackground() != grayCode) {
      cell.setFormula(weeklyFormula);
    }
    
    console.log(dataCol+rowNum);
    rowNum++;
  }
}

/**
 * get the date of the week
 */
function getDate(sheet) {
  let newRawDate = sheet.getRange(weeklyCol+1).getValue();
  let newDate = getDateString(newRawDate);
  return newDate;
}

/*
 * This is the helper function to convert the readin date into the
 * correct format. The read in date would be presented as
 * Sun Jan 09 2022 03:00:00 GMT-0500 (Eastern Standard Time)
 * but we want 01/09/2022
 */
function getDateString(date) {
  let dateArr = date.toString().split(" ");
  let month = dateArr[1];
  let dateString;

  // a for loop to loop the kv array
  // note: wants to use map at the beginning but it's kinda hard to use in this language
  let i = 0;
  for(i = 0; i < monthsMap.length; i++) {
    if (monthsMap[i].month == month) {
      dateString = monthsMap[i].val;
      break;
    }
  }
  dateString += '/'+ dateArr[2] + '/' + dateArr[3][2] + dateArr[3][3];
  return dateString;
}

/**
 * a helper function for debugging 
 */
function getRightFormula(sheet) {
  return sheet.getRange('E3').getFormula();
}

/**
 * the only function needs to be executed 
 */
function main() {
  // apply contribution
  console.log("contribution");
  let sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
  console.log(sheet.getRange(1,1).getValue());
  let date = getDate(sheet);
  applyFormula(sheet, sheetNameArr[0], date);
  
  // apply benev
  console.log("benev");
  sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[1];
  date = getDate(SpreadsheetApp.getActiveSpreadsheet().getSheets()[3]);
  applyFormula(sheet, sheetNameArr[1], date);

  // apply mission
  console.log("mission");
  sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[2];
  date = getDate(sheet);
  applyFormula(sheet, sheetNameArr[2], date);

  // apply miscellaneous
  console.log("miscellaneous");
  sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[3];
  date = getDate(sheet);
  applyFormula(sheet, sheetNameArr[3], date);
}
