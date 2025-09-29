/** Common base for public market data clients. */
export abstract class BaseExchangePublic {
  protected readonly name: string = 'exchange';
  protected now(): number { return Date.now(); }

  // Interfaces to implement
  abstract getTicker(pair: string): Promise<any>;
  abstract getOrderBook(pair: string): Promise<any>;
  abstract getTrades(pair: string): Promise<any>;
}
