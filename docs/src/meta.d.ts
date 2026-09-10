/** The frontmatter of every page under content/, read at build by ../meta.ts. */
declare module "virtual:meta" {
  const pages: Record<string, Record<string, string>>;
  export default pages;
}
