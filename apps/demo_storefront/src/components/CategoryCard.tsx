import type { Category } from "../data/products";

interface CategoryCardProps {
  category: Category;
  onChoose: (category: Category) => void;
}

export function CategoryCard({ category, onChoose }: CategoryCardProps): JSX.Element {
  return (
    <button
      type="button"
      className="category-card"
      onClick={() => onChoose(category)}
      style={{ background: `linear-gradient(145deg, ${category.accent}, #fff 70%)` }}
      aria-label={`Browse ${category.name}`}
    >
      <p className="category-subtitle">{category.subtitle}</p>
      <h3>{category.name}</h3>
      <p>{category.note}</p>
      <span className="category-cta">Open category</span>
    </button>
  );
}
