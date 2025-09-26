import * as pd from 'pandas-js';

export class CommonPreprocessor {
  static preprocess(data: any): any {
    // Basic preprocessing: ensure date column is in proper format
    if (data['date']) {
      data['date'] = data['date'].map((d: any) => new Date(d));
    }
    // Add any other common preprocessing here
    return data;
  }
}
